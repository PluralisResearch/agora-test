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

import random
import time

from contextlib import nullcontext
from typing import Any

import torch

from agora_server.core.coordination.training_coordinator import TrainingCoordinator
from agora_server.hivemind.moe.server import ModuleBackend
from agora_server.hivemind.utils.logging import get_logger
from agora_server.hivemind.utils.nested import nested_compare, nested_flatten, nested_pack


logger = get_logger(__name__)


def _get_autocast_context(use_mixed_precision: bool, device: torch.device):
    """Return autocast context manager for BF16 mixed precision, or nullcontext if disabled."""
    if use_mixed_precision and device.type == "cuda":
        # NOTE: FP16 (which requires GradScaler) are NOT safe, don't use it here
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


class ModuleCollab(ModuleBackend):
    # Narrows ModuleBackend's generic torch optimizer: backward here drives the collaborative
    # step(batch_size=...) contract. Assigned by server.create after construction.
    optimizer: TrainingCoordinator | None

    def __init__(
        self,
        optimizer_lock: Any,
        report_interval_minutes: float = 1,
        delay_range_forward_backward: tuple[float, float] = (0.0, 0.0),
        use_mixed_precision: bool = False,
        use_torch_compile: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.optimizer_lock = optimizer_lock
        self.delay_range_forward_backward = delay_range_forward_backward
        self.use_mixed_precision = use_mixed_precision
        self.use_torch_compile = use_torch_compile

        if use_torch_compile:
            self._compiled_module = torch.compile(self.module, dynamic=True)
            logger.info("torch.compile enabled (regional: forward/backward call-sites only)")

        # Gradient reporting configuration
        self.report_interval_minutes = report_interval_minutes
        self.report_interval_seconds = report_interval_minutes * 60
        self.last_report_time = time.time()
        self.grad_norms = []  # Store accumulated gradient norms

        if self.use_mixed_precision:
            logger.info("Mixed precision (BF16) enabled for forward/backward passes")

    def backward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply backward pass to an aggregated batch of requests.

        Used by Runtime, do not call this manually.
        To submit a request for asynchronous processing, please use ``ModuleBackend.backward_pool.submit_task``.

        Subclassing:
           - This method receives a sequence of torch tensors following ``nested_flatten(self.backward_schema)``;

           - It should return gradients w.r.t. inputs that follow ``nested_flatten(self.forward_schema)``;

           - Runtime doesn't guarantee that backward will be performed in the same order and for the same data
           as forward, so we recommend stateless backward pass that re-runs expert forward pass inside backward.

           - Please make sure to call ``ModuleBackend.on_backward`` after each call to backward
        """
        (args, kwargs), grad_outputs = nested_pack(inputs, structure=self.backward_schema)

        with torch.enable_grad():
            with self.optimizer_lock:
                args = [
                    tensor.detach().requires_grad_(True) if tensor.is_floating_point() else tensor.detach()
                    for tensor in args
                ]
                kwargs = {
                    input_key: (
                        tensor.detach().requires_grad_(True) if tensor.is_floating_point() else tensor.detach()
                    )
                    for input_key, tensor in kwargs.items()
                }

                batch_size = args[0].size(0)
                device = args[0].device

                # Use BF16 autocast for forward pass only (per PyTorch AMP recommendations)
                # Backward ops automatically run in the same dtype as their corresponding forward ops
                # param.grad is always FP32 (matches param.dtype)
                with _get_autocast_context(self.use_mixed_precision, device):
                    _module = self._compiled_module if self.use_torch_compile else self.module
                    outputs = _module(*args, **kwargs)

                assert nested_compare(outputs, grad_outputs), "outputs and grad_outputs must have the same structure"

                outputs_flat = tuple(nested_flatten(outputs))

                grad_outputs_flat = tuple(
                    map(
                        lambda grad, out: grad.to(device=out.device, dtype=out.dtype, non_blocking=True),
                        nested_flatten(grad_outputs),
                        outputs_flat,
                    )
                )

                torch.autograd.backward(
                    outputs_flat, grad_tensors=grad_outputs_flat, create_graph=False, retain_graph=False
                )

            self.on_backward(batch_size)

        # Delay used to simulate latency
        if self.delay_range_forward_backward[1] > 0:
            time.sleep(random.uniform(self.delay_range_forward_backward[0], self.delay_range_forward_backward[1]))

        return tuple(
            x.grad if isinstance(x.grad, torch.Tensor) else torch.zeros_like(x) for x in nested_flatten((args, kwargs))
        )

    def on_backward(self, batch_size: int) -> None:
        """Train the expert for one step.

        This method is called by ``ModuleBackend.backward`` after computing gradients.
        """
        # Logging grad norms
        grads = [p.grad for p in self.module.parameters() if p.grad is not None]

        # Safety check: gradients must be FP32 for stable accumulation and optimizer step
        # FP16 gradients risk underflow - only BF16 autocast (same dynamic range as FP32) is safe.
        if grads:
            assert grads[0].dtype == torch.float32, f"Expected FP32 gradients, got {grads[0].dtype}"

        # Keep grad norms on device to avoid sync
        self.grad_norms.append(torch.nn.utils.get_total_norm(grads).detach())

        if self.optimizer is not None:
            self.optimizer.step(batch_size=batch_size)

            # With reuse_grad_buffers=True the optimizer accumulates into param.grad across steps,
            # so the model gradients must not be cleared here.
            if self.optimizer.should_zero_grad_after_backward():
                self.optimizer.zero_grad(set_to_none=not self.use_torch_compile)

            if self.scheduler is not None:
                self.scheduler.step()

        self._maybe_report_grad_stats()

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Overrides `ModuleBackend.forward` to add a delay and optional mixed precision."""
        # Delay used to simulate latency
        if self.delay_range_forward_backward[1] > 0:
            time.sleep(random.uniform(self.delay_range_forward_backward[0], self.delay_range_forward_backward[1]))

        # Determine device from first input tensor
        device = inputs[0].device if inputs else torch.device("cpu")

        # Use BF16 autocast for forward pass if mixed precision is enabled
        with self.optimizer_lock:
            with _get_autocast_context(self.use_mixed_precision, device):
                if self.use_torch_compile:
                    args, kwargs = nested_pack(inputs, structure=self.forward_schema)
                    with torch.no_grad():
                        outputs = self._compiled_module(*args, **kwargs)
                    return tuple(nested_flatten(outputs))
                return super().forward(*inputs)

    def _maybe_report_grad_stats(self):
        """Check if enough time has passed and report gradient statistics if so."""
        current_time = time.time()

        if current_time - self.last_report_time >= self.report_interval_seconds:
            if self.grad_norms:
                norms = torch.stack(self.grad_norms)
                max_grad = norms.max().item()
                avg_grad = norms.mean().item()

                logger.info(f"Pre Max total_grad: {max_grad:.6f}")
                logger.info(f"Pre Avg total_grad: {avg_grad:.6f}")

                # Reset for next interval
                self.grad_norms.clear()
            else:
                logger.info(f"No gradient norms recorded in the last {self.report_interval_minutes} minutes")

            self.last_report_time = current_time

    def get_info(self) -> dict[str, Any]:
        """Get expert parameters and stats. Used by RemoteExpert to check shapes and for DMoE orchestration."""
        info = super().get_info()
        if hasattr(self.module, "model_args"):
            info["model_args"] = dict(self.module.model_args)

        return info


class TailModuleCollab(ModuleCollab):
    """Fused-forward variant of ModuleCollab for the lm_tail expert.

    The tail is the last stage of the pipeline — there is no downstream stage waiting on
    its forward output, so the standard recompute-in-backward pattern wastes one full
    forward per step. This subclass folds the backward pass into the forward RPC:

    Forward inputs (from the trainer-side fused shim):
        (hidden, labels, loss_weight)
        - hidden: (B, T, H) float — kept under the autograd graph and given .grad
        - labels: (B, T) long — token targets
        - loss_weight: (B, T) float — per-token weight.
            * Training: trainer sets ones / (B*T). Under TaskPool coalescing, the merged
              batch carries heterogeneous per-trainer weights, which makes
              (loss * loss_weight).sum() reproduce the per-trainer mean-reduction that
              today's two-RPC path achieves via per-request grad_outputs.
            * Eval / no-update: trainer sets all-zeros to signal "compute loss only,
              do NOT update parameters". The forward path detects this and skips
              backward + on_backward, so the worker does not mutate state during eval.

    Forward outputs:
        (loss_per_token, grad_hidden) — both batch-first, dim-0 splittable by Runtime.

    The trainer-side fused shim consumes (loss_per_token, grad_hidden) from the forward
    RPC and never issues a backward RPC. `backward` is overridden to raise with a clear
    misconfiguration message: any backward call against a fused-tail backend would have
    a 3-input forward schema that's incompatible with the underlying TailExpert's 2-arg
    forward, so the ModuleCollab.backward recompute path cannot serve as a transparent
    fallback. Loud failure is preferable to silent corruption.
    """

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Delay used to simulate latency
        if self.delay_range_forward_backward[1] > 0:
            time.sleep(random.uniform(self.delay_range_forward_backward[0], self.delay_range_forward_backward[1]))

        args, kwargs = nested_pack(inputs, structure=self.forward_schema)
        assert not kwargs, (
            "TailModuleCollab does not use kwargs; tail expert takes positional (hidden, labels, loss_weight)"
        )
        assert len(args) == 3, (
            f"TailModuleCollab expects (hidden, labels, loss_weight); got {len(args)} positional inputs"
        )

        hidden_in = args[0]
        labels = args[1].detach()
        loss_weight_in = args[2].detach()
        device = hidden_in.device

        # Eval / no-update path. The trainer signals "do not update worker params" by
        # setting loss_weight = zeros. Skipping backward + on_backward here is what
        # makes evaluation safe — otherwise every forward call mutates the worker.
        # We compute loss_per_token under no_grad and return zero grad_hidden.
        if not torch.any(loss_weight_in):
            with torch.no_grad():
                with _get_autocast_context(self.use_mixed_precision, device):
                    _module = self._compiled_module if self.use_torch_compile else self.module
                    loss_per_token = _module(hidden_in.detach(), labels)
            return (loss_per_token.detach(), torch.zeros_like(hidden_in))

        # Training path.
        hidden = hidden_in.detach().requires_grad_(True)
        batch_size = hidden.size(0)

        with torch.enable_grad():
            with self.optimizer_lock:
                # Forward under autocast (matches today's backward path).
                # Backward ops automatically run in the same dtype as their corresponding forward ops;
                # param.grad is FP32 (matches param.dtype).
                with _get_autocast_context(self.use_mixed_precision, device):
                    _module = self._compiled_module if self.use_torch_compile else self.module
                    loss_per_token = _module(hidden, labels)

                # Move loss_weight to the loss tensor's device/dtype so the multiply broadcasts cleanly.
                loss_weight = loss_weight_in.to(
                    device=loss_per_token.device, dtype=loss_per_token.dtype, non_blocking=True
                )
                weighted_loss = (loss_per_token * loss_weight).sum()
                weighted_loss.backward()

            # Read grad w.r.t. hidden, then drop the autograd graph.
            # Lock is released here — on_backward must run outside the lock because
            # SPARTA optimizer.step re-acquires the same non-reentrant mp.Lock.
            grad_hidden = hidden.grad.detach() if hidden.grad is not None else torch.zeros_like(hidden)

            self.on_backward(batch_size)

        return (loss_per_token.detach(), grad_hidden)

    def backward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        raise RuntimeError(
            "TailModuleCollab.backward was invoked, but the fused-tail backend serves all "
            "training via its forward RPC. A backward RPC indicates a client/server config "
            "mismatch: the trainer is using a non-fused remote-expert wrapper "
            "(BalancedRemoteExpert) against a fused-tail worker (TailModuleCollab). Either "
            "switch the trainer to BalancedRemoteFusedTailExpert or run the worker with "
            "plain ModuleCollab."
        )
