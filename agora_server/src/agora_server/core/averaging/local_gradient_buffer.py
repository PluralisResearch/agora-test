# Copyright 2026 Pluralis Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contextlib

from typing import Iterable, Iterator, Sequence  # noqa: UP035

import torch

from agora_server.core.averaging.gradient_buffer import GradientBuffer
from agora_server.hivemind.utils import get_logger


logger = get_logger(__name__)


class LocalGradientBuffer(GradientBuffer):
    """Local gradient accumulator with no peer all-reduce.

    This class manages two copies of the gradients:

    1. accumulated gradients - raw per-micro-batch sums, on the parameter device (GPU).
       If ``reuse_grad_buffers`` is True these alias ``param.grad`` directly.
    2. averaged gradients - the per-sample mean, in host (CPU) memory, produced by
       ``populate_averaged_grads_`` and returned by ``get_averaged_grads``.
    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        *,
        reuse_grad_buffers: bool = True,
        accumulate_grads_on: torch.device | None = None,
        averaged_grads: Sequence[torch.Tensor] = (),
    ):
        """Initialize a local gradient buffer.

        Args:
            parameters (Iterable[torch.nn.Parameter]): Parameters whose gradients to accumulate.
            reuse_grad_buffers (bool, optional): If True, use the model's ``.grad`` buffers as the
                accumulators (memory efficient, but the caller must not call zero_grad/clip_grad
                manually). Defaults to False.
            accumulate_grads_on (torch.device | None, optional): If set, accumulate gradients on this
                device (e.g. 'cpu') to save device memory at the cost of extra copies. No effect if
                ``reuse_grad_buffers`` is True. Defaults to None (parameter device).
            averaged_grads (Sequence[torch.Tensor], optional): Pre-allocated host averaged-gradient
                buffers to use instead of allocating new ones. Defaults to ().
        """
        if reuse_grad_buffers and accumulate_grads_on is not None:
            logger.warning("Setting 'accumulate_grads_on' has no effect if reuse_grad_buffers=True")

        self.parameters = tuple(parameters)
        self.reuse_grad_buffers = reuse_grad_buffers
        self.local_samples_accumulated = 0
        self.local_times_accumulated = 0
        self._local_accumulators = None

        if not reuse_grad_buffers:
            self._local_accumulators = tuple(
                torch.zeros_like(grad, device=accumulate_grads_on) for grad in self._grads_from_parameters()
            )

        with torch.no_grad():
            if not averaged_grads:
                # Host (CPU) buffers, always in host memory - aliased onto the offloaded optimizer's .grad.
                self._averaged_grads = tuple(
                    grad.detach().cpu().clone().share_memory_() for grad in self._grads_from_parameters()
                )
            else:
                if any(
                    param_grad.size() != grad.size()
                    for param_grad, grad in zip(self._grads_from_parameters(), averaged_grads)
                ):
                    raise ValueError("Averaged gradients don't have same shape as gradients from parameters")
                self._averaged_grads = tuple(averaged_grads)

    def _grads_from_parameters(self) -> Iterator[torch.Tensor]:
        """Gradient buffers associated with parameters."""
        for param in self.parameters:
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            yield param.grad

    @torch.no_grad()
    def _grads_from_accumulators(self) -> Iterator[torch.Tensor]:
        """Buffers in which local gradients are accumulated across micro-batches."""
        if self.reuse_grad_buffers:
            yield from self._grads_from_parameters()
        else:
            assert self._local_accumulators is not None
            yield from self._local_accumulators

    @contextlib.contextmanager
    def get_averaged_grads(self):
        """Yield the host (CPU) averaged-gradient buffers (aliased onto the offloaded optimizer .grad)."""
        yield self._averaged_grads

    @torch.no_grad()
    def accumulate_grads_(self, batch_size: int):
        """Add current gradients to the accumulated (sum) buffers.

        Gradients from torch.autograd are sum-reduced over the batch dimension; we keep raw sums and
        divide by total samples in :meth:`populate_averaged_grads_` to get the per-sample mean.
        """
        self.local_samples_accumulated += batch_size
        self.local_times_accumulated += 1
        if self.reuse_grad_buffers:
            return  # caller is responsible for accumulating gradients in the .grad buffers
        for grad_buf, grad_acc in zip(self._grads_from_parameters(), self._grads_from_accumulators()):
            grad_acc.add_(grad_buf.to(grad_acc.device))

    @torch.no_grad()
    def populate_averaged_grads_(self):
        """Reduce the accumulated sums to the per-sample mean and write it into the averaged buffers."""
        grad_scale = (1.0 / self.local_samples_accumulated) if self.local_samples_accumulated != 0 else 0.0
        with self.get_averaged_grads() as averaged_grads:
            for grad_acc, averaged_grad in zip(self._grads_from_accumulators(), averaged_grads):
                averaged_grad.copy_(grad_acc, non_blocking=True).mul_(grad_scale)

    @torch.no_grad()
    def reset_accumulated_grads_(self):
        """Zero the accumulated buffers and the counters."""
        self.local_samples_accumulated = self.local_times_accumulated = 0
        for grad_buf in self._grads_from_accumulators():
            grad_buf.zero_()

    def has_nan_grads(self) -> bool:
        """Check if any gradient contains NaN or inf values."""
        flags = [torch.isfinite(grad).all() for grad in self._grads_from_parameters()]
        return bool(flags) and not bool(torch.stack(flags).all())
