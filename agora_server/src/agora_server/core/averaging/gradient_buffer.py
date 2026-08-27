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

from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import torch


GradientBufferFactory = Callable[..., "GradientBuffer"]


@runtime_checkable
class GradientBuffer(Protocol):
    """The interface for local gradient accumulation and averaged-gradient access.

    An implementation manages two copies of the gradients:

    1. accumulated gradients - raw per-micro-batch sums, on the parameter device (GPU). If
       ``reuse_grad_buffers`` is True these alias ``param.grad`` directly.
    2. averaged gradients - the per-sample mean, in host (CPU) memory, produced by
       ``populate_averaged_grads_`` and returned by ``get_averaged_grads``.
    """

    reuse_grad_buffers: bool
    local_samples_accumulated: int

    @abstractmethod
    def accumulate_grads_(self, batch_size: int) -> None:
        """Add the current ``param.grad`` values into the accumulated (sum) buffers."""

    @abstractmethod
    def populate_averaged_grads_(self) -> None:
        """Reduce the accumulated sums to the per-sample mean and write it into the averaged buffers."""

    @abstractmethod
    def reset_accumulated_grads_(self) -> None:
        """Zero the accumulated buffers and the sample/step counters."""

    @abstractmethod
    def get_averaged_grads(self) -> contextlib.AbstractContextManager[Sequence[torch.Tensor]]:
        """Context manager yielding the host (CPU) averaged-gradient buffers."""

    @abstractmethod
    def has_nan_grads(self) -> bool:
        """Whether any current gradient contains NaN or inf."""
