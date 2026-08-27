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
"""Interface of the training coordinators used across agora_server."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from agora_server.types import GhostPhase, ParamGroups, TorchOptimizer


if TYPE_CHECKING:
    from agora_server.core.coordination.progress_tracker import ProgressTracker


class TrainingCoordinator(torch.optim.Optimizer, ABC):
    """Interface every training coordinator implements.

    A training coordinator wraps a regular PyTorch optimizer (exposed as :attr:`opt`) and coordinates it with the
    swarm: it accumulates local gradients, tracks global training progress, and keeps parameters and optimizer state
    synchronized with peers. Components such as the server, the module backends, and the gradient scaler rely only on
    the members declared here, so implementations are interchangeable regardless of how they schedule updates
    (synchronous or asynchronous) or which averaging strategy they use.

    Training time is measured in *epochs*: one epoch corresponds to the swarm collectively processing a target number
    of samples, after which peers transition together (apply an optimizer update, advance the learning-rate scheduler,
    average state). An epoch is not a pass over the training data. Peers may join or leave mid-run, so any
    time-dependent behavior (schedulers, curriculum) must be driven by :attr:`local_epoch`, never by the number of
    local ``step`` calls.

    Lifecycle: construct, then ``load_state_from_peers`` (unless seeding the run or restoring from a checkpoint),
    then ``start_monitor``, then repeated ``step`` calls from backward passes, and finally ``shutdown``.

    This class inherits ``torch.optim.Optimizer`` for ecosystem compatibility (AMP gradient scaling, scheduler
    factories, ``isinstance`` checks) but never calls its ``__init__``; implementations manage parameters through
    their own machinery. Ghost-mode members (``ghost_phase``, ``ghost_phase_start_epoch``, ``can_checkpoint``,
    ``wait_for_join_window``) are optional capabilities with neutral defaults; implementations without ghost support
    inherit them as-is.

    Attributes:
        tracker (ProgressTracker): Local and global training progress tracker; implementations must create it during
            construction. Consumers may read it and toggle its ``allow_progress_report``.
        ghost_phase_start_epoch (int | None): Epoch at which the current ghost phase started; None when ghost mode
            is off or unsupported.
    """

    tracker: ProgressTracker
    ghost_phase_start_epoch: int | None = None

    # step and param_groups deliberately shadow torch.optim.Optimizer's incompatible declarations;
    # torch's __init__ never runs and its step(closure) contract does not apply (see class docstring).
    @abstractmethod
    def step(self, batch_size: int | None = None) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Report a locally processed batch and run collaborative optimization once the swarm is ready.

        Unlike ``torch.optim.Optimizer.step``, this is called after every backward pass with the local batch size;
        the actual parameter update happens only when peers collectively accumulate the target batch size.

        Args:
            batch_size (int | None, optional): Number of samples processed since the previous call. Defaults to the
                implementation's configured per-step batch size.
        """

    @abstractmethod
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Reset gradients of all optimized parameters."""

    @abstractmethod
    def should_zero_grad_after_backward(self) -> bool:
        """Whether the training loop must clear model gradients after each backward pass."""

    @abstractmethod
    def state_dict(self) -> dict:
        """Return optimizer state for checkpointing, including the local epoch."""

    @abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        """Restore optimizer state produced by :meth:`state_dict`."""

    @abstractmethod
    def load_state_from_peers(self, wait_for_end_round: bool = False, **kwargs) -> None:
        """Download the newest collaboration state (parameters, optimizer state, epoch) from peers, in-place.

        Args:
            wait_for_end_round (bool, optional): If True, additionally wait for the swarm to begin a fresh
                accumulation round before returning. Defaults to False.
            **kwargs: Forwarded to the state averager's ``load_state_from_peers``.
        """

    @abstractmethod
    def set_averagers_allow_state_sharing(self) -> None:
        """Apply the configured ``allow_state_sharing`` policy to the underlying averagers."""

    @abstractmethod
    def start_monitor(self) -> None:
        """Start the background watchdog that keeps collaborative progress going between external ``step`` calls."""

    @abstractmethod
    def stop_monitoring(self) -> None:
        """Stop the background watchdog started by :meth:`start_monitor`."""

    @abstractmethod
    def is_alive(self) -> bool:
        """Whether the coordinator's background components are still running."""

    @abstractmethod
    def shutdown(self) -> None:
        """Stop background threads and averagers and notify peers; the coordinator is unusable afterwards."""

    @property
    @abstractmethod
    def local_epoch(self) -> int:
        """This peer's current epoch, kept synchronized with the swarm."""

    @property
    @abstractmethod
    def opt(self) -> TorchOptimizer:
        """The wrapped PyTorch optimizer that applies parameter updates."""

    @property
    @abstractmethod
    def param_groups(self) -> ParamGroups:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Parameter groups of the wrapped optimizer, viewed over the main (averaged) parameters."""

    @property
    def ghost_phase(self) -> GhostPhase:
        """Current ghost-mode phase; OFF for implementations without ghost support."""
        return GhostPhase.OFF

    @property
    def can_checkpoint(self) -> bool:
        """Whether this coordinator is in a state that can be checkpointed."""
        return True

    def wait_for_join_window(self) -> None:
        """Block until it is safe to join the run; no-op for implementations without join windows."""

    def add_param_group(self, param_group: dict) -> None:
        """Unsupported: training coordinators require all parameter groups at construction time."""
        raise ValueError(
            f"{self.__class__.__name__} does not support calling add_param_group after creation. "
            f"Please provide all parameter groups at init"
        )
