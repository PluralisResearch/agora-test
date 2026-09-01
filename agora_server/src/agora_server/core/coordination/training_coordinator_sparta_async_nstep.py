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
import time

from typing import Callable  # noqa: UP035

import torch

from agora_server.core.coordination.training_coordinator_base import TrainingCoordinatorBase
from agora_server.hivemind.optim.grad_scaler import GradScaler
from agora_server.hivemind.utils import get_dht_time, get_logger
from agora_server.types import GhostPhase


logger = get_logger(__name__)


class TrainingCoordinatorAsyncSparta(TrainingCoordinatorBase):
    """
    Asynchronous N-step delay SPARTA training coordinator.

    Training Flow:
    - Peers accumulate gradients locally and take optimizer steps when target batchsize is met
    - Every N steps, peers trigger weighted state averaging in background
    - While state averaging runs asynchronously, training continues with next N local steps
    - When averaging completes, it interrupts training to apply averaged parameters to GPU

    Note:
        This implementation assumes reuse_tensors=False to maintain separate buffers
        for local optimization and peer averaging, enabling true N-step delay behavior.

    Ghost mode (two-phase):
        Ghost mode allows a peer that joins with stale weights to catch up without negatively
        impacting the swarm. It has two phases (see GhostPhase enum):

        Phase 1 (receive-only): Worker is invisible to trainers — no batches are sent to it.
            It participates in state AR with weight=0, receiving averaged weights from peers.
            Enabled by setting ghost_phase1_steps > 0.

        Phase 2 (warm-up): Worker becomes visible to trainers and processes real batches (FW/BW),
            warming up optimizer momentum/variance with real gradients. It still does not contribute
            to AR (weight=0) or progress tracking (reports 0 samples). Enabled by setting
            ghost_phase2_steps > 0. Phase 2 only activates after Phase 1 completes.

        Worker enters ghost mode if the downloaded state is more than `max_allowed_stale` epochs behind.
        Ghost phase is communicated to trainers via DHT (see `declare_experts` in dht_handler.py).
    """

    def __init__(
        self,
        ghost_phase1_steps: int = 0,
        ghost_phase2_steps: int = 0,
        load_state_block_window: int = 0,
        *args,
        **kwargs,
    ):
        """
        Initialize the TrainingCoordinatorAsyncSparta.
        Args:
            ghost_phase1_steps (int): Number of local steps for ghost phase 1 (receive-only, no batches).
                Defaults to 0, meaning ghost mode is disabled.
            ghost_phase2_steps (int): Number of local steps for ghost phase 2 (process batches, warm up optimizer,
                still weight=0 in AR). Defaults to 0. Only takes effect if ghost_phase1_steps > 0.
            load_state_block_window (int): If > 0, block loading state from peers if global epoch
                is within this many steps of an averaging round. Defaults to 0. Can't be greater than half of average_state_every.
        """
        super().__init__(*args, **kwargs)
        assert self.grad_averager is None and (self.grad_buffer is not None or self.auxiliary), (
            "TrainingCoordinatorAsyncSparta does not support grad_averager, use grad_buffer instead; auxiliary peers run with neither."
        )

        self.local_samples_accumulated = 0  # Accumulating the samples across the steps

        assert load_state_block_window * 2 < self.average_state_every, (
            "load_state_block_window should be less than half of average_state_every"
        )
        self.load_state_block_window = load_state_block_window
        self.ghost_phase1_steps = ghost_phase1_steps
        self.ghost_phase2_steps = ghost_phase2_steps
        self._ghost_phase = GhostPhase.OFF
        self.ghost_phase_start_epoch = None

        if self.auxiliary:
            self.allow_state_sharing = False
            self._enter_ghost_mode()

    def is_ghost_mode_supported(self):
        """
        Returns true if workers are allowed to use ghost mode (i.e. ghost_phase1_steps > 0),
        or if this is an auxiliary peer (a reducer lives permanently in ghost phase 1).
        """
        return self.ghost_phase1_steps > 0 or self.auxiliary

    def _resync_state(self):
        """
        Never reload weights from peers, always catch up if behind, use ghost mode if supported.
        Note:
            This should be called after loading a potentially stale state (e.g. from peers or checkpoint)
            to ensure ghost mode is handled appropriately.
        """
        # check whether to enter ghost mode (phase 1)
        if self.is_ghost_mode_supported() and self.ghost_phase == GhostPhase.OFF:
            gap = self.tracker.global_epoch - self.local_epoch
            if gap > self.max_allowed_stale:
                logger.info(
                    f"Entering ghost phase 1 at epoch {self.tracker.global_epoch}, state is {gap} epochs stale."
                )
                self._enter_ghost_mode()
                logger.info(
                    f"Ghost phase 1 will last {self.ghost_phase1_steps} steps, "
                    f"ending at local epoch {self.ghost_phase_start_epoch + self.ghost_phase1_steps}"
                )

        # catch up if behind
        if self._catchup_epoch():
            with self.tracker.pause_updates():
                logger.log(self.status_loglevel, f"Catching up with collaboration step {self.tracker.global_epoch}")
                self.state_averager.local_epoch = self.tracker.global_epoch
                self.tracker.report_local_progress(local_epoch=self.local_epoch, samples_accumulated=0)

        # check ghost phase transitions
        if self.ghost_phase != GhostPhase.OFF:
            logger.debug(
                f"In ghost {self.ghost_phase} since epoch #{self.ghost_phase_start_epoch}, "
                f"now is #{self.local_epoch} local, #{self.tracker.global_epoch} global"
            )
            self._maybe_transition_ghost_phase()

        return False

    def _maybe_transition_ghost_phase(self):
        if self.auxiliary:
            return
        if self.ghost_phase == GhostPhase.PHASE1:
            if self.local_epoch >= self.ghost_phase_start_epoch + self.ghost_phase1_steps:
                if self.ghost_phase2_steps > 0:
                    self._transition_to_phase2()
                    logger.info(f"Transitioning to ghost phase 2 at {self.local_epoch} local epoch")
                    logger.info(
                        f"Ghost phase 2 will last {self.ghost_phase2_steps} steps, "
                        f"ending at local epoch {self.ghost_phase_start_epoch + self.ghost_phase2_steps}"
                    )
                else:
                    self._exit_ghost_mode()
                    logger.info(f"Exiting ghost mode at {self.local_epoch} local epoch")

        elif self.ghost_phase == GhostPhase.PHASE2:
            if self.local_epoch >= self.ghost_phase_start_epoch + self.ghost_phase2_steps:
                self._exit_ghost_mode()
                logger.info(f"Exiting ghost mode at {self.local_epoch} local epoch")

    def _should_load_state_from_peers(self) -> bool:
        """
        Never reload weights from peers, use ghost mode if behind.
        """
        return False

    def _catchup_epoch(self) -> bool:
        """
        Only catch up if behind by 2+ steps - because behind by 1 will trigger averaging.
        """
        gap = self.tracker.global_epoch - self.local_epoch
        if gap > self.max_allowed_stale and not self.is_ghost_mode_supported():
            logger.warning(
                f"Local epoch {self.local_epoch} is behind global epoch {self.tracker.global_epoch} by {gap} epochs, "
                f"but ghost mode is disabled. Catching up by skipping to global epoch."
            )
        return gap > 1

    def _auto_step(self) -> None:
        """Internal method to call `step()` automatically."""
        try:
            # Call the parent class's step method with one batch size
            logger.debug(f"Auto step at {time.strftime('%H:%M:%S')}")
            self._last_step_time = time.time()

            # if delayed updates finished before step, apply these updates; otherwise do nothing
            with self.optimizer_lock:
                self.state_averager.step(apply_delayed_updates=True)

            if self._resync_state():
                return None

            self._maybe_start_optimizer_step()

            if self.in_update and self.tracker.ready_to_update_epoch:
                self._step(batch_size=None if self.auxiliary else 1)

            self.state_averager.allow_state_sharing = self.allow_state_sharing
        except Exception as e:
            logger.exception(f"Error in auto step: {e}")

    def _step(
        self,
        closure: Callable[[], torch.Tensor] | None = None,
        batch_size: int | None = None,
        grad_scaler: GradScaler | None = None,
    ):
        """Update training progress after accumulating another local batch size.

        Depending on the configuration, this will report progress to peers, run global or local optimizer step, average parameters or schedule background tasks.

        Args:
            closure (Callable[[],torch.Tensor] | None, optional): A closure that reevaluates the model and returns the loss. Defaults to None.
            batch_size (int | None, optional): Optional override for batch_size_per_step from init. Defaults to None.
            grad_scaler (GradScaler | None, optional): Gradient scaler. Defaults to None.

        Raises:
            ValueError: If grad_scaler is not a GradScaler instance, if batch_size and batch_size_per_step are not provided and auxiliary mode is disabled, or if closure/batch_size are provided in auxiliary mode.

        Note:
            This step method differs from standard PyTorch optimizers in several key ways. See __init__ for details.
        """
        if grad_scaler is not None and not isinstance(grad_scaler, GradScaler):
            raise ValueError("hivemind.Optimizer requires a hivemind-aware gradient scaler (hivemind.GradScaler)")
        if self.batch_size_per_step is None and batch_size is None and not self.auxiliary:
            raise ValueError("Please either set batch_size_per_step parameter at init or when calling .step")
        if self.auxiliary and (closure is not None or batch_size is not None):
            raise ValueError("Auxiliary peers should not have batch size, run closures, or use grad_scaler")
        batch_size = batch_size if batch_size is not None else self.batch_size_per_step

        # if delayed updates finished before step, apply these updates; otherwise do nothing
        with self.optimizer_lock:
            self.state_averager.step(apply_delayed_updates=True)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Phase 1: skip gradient accumulation entirely (no batches expected)
        if self.ghost_phase != GhostPhase.PHASE1:
            override_samples_accumulated = 0 if self.ghost_phase == GhostPhase.PHASE2 else None
            grads_are_valid = self._check_and_accumulate_gradients(
                batch_size, grad_scaler, override_samples_accumulated=override_samples_accumulated
            )
            if not grads_are_valid:
                return loss  # local gradients were reset due to overflow, must start over

        self._maybe_start_optimizer_step()

        if self.in_update and self.tracker.ready_to_update_epoch:
            self.state_averager.allow_state_sharing = False  # Prevent state sharing during AR.
            with self.optimizer_lock:
                self._update_global_epoch(grad_scaler)

            # Auxiliary peers release their non-averaged weights at startup; there is nothing to regularize.
            if self.model.model_args.use_compression and not self.auxiliary:
                self.model.ss_regularize()

            self.in_update = False
        return loss

    def _update_global_epoch(self, grad_scaler: GradScaler | None) -> None:
        # 1) Load local accumulated gradients into optimizer buffer
        # 2) Take a local step
        # 3) Every N local steps perform a weighted sparta state average
        assert self._schema_hash == self._compute_schema_hash(), "parameters or gradients changed during iteration"

        with self.tracker.pause_updates():
            next_epoch = max(self.local_epoch + 1, self.tracker.global_epoch)
            self.state_averager.global_epoch = next_epoch - 1

            logger.log(self.status_loglevel, f"Beginning optimizer step #{self.local_epoch}")
            if self.grad_buffer is not None:
                self.local_samples_accumulated += self.grad_buffer.local_samples_accumulated
                self.grad_buffer.populate_averaged_grads_()
                self.grad_buffer.reset_accumulated_grads_()
                self._load_averaged_gradients_into_optimizer_()
            # Without gradients this only advances the epoch and the scheduler.
            self.state_averager.local_optimizer_step(increment_epoch=True)
            if self.grad_buffer is not None and self.grad_buffer.reuse_grad_buffers:
                self.grad_buffer.reset_accumulated_grads_()

            swarm_not_empty = self.tracker.global_progress.num_peers > 1
            should_average_state = (
                swarm_not_empty
                and next_epoch % self.average_state_every == 0
                and not self.state_averager.averaging_in_progress
            )

            if should_average_state:
                logger.log(self.status_loglevel, f"Launching async state averaging for step #{self.local_epoch}")
                self.state_averager.step(wait_for_delayed_updates=True)
                self.state_averager.step(
                    increment_epoch=False,
                    wait_for_trigger=None,
                    optimizer_step=False,
                    delay_optimizer_step=False,
                    grad_scaler=grad_scaler,
                    averaging_round=True,
                    delay_averaging=True,
                    averaging_control=None,
                    averaging_opts=dict(
                        timeout=self.averaging_timeout,
                        weight=0 if self.ghost_phase != GhostPhase.OFF else self.local_samples_accumulated,
                    ),
                )
                self.local_samples_accumulated = 0  # Reset every state average

            if not should_average_state and self.scheduled_state is not None and not self.scheduled_state.done():
                self.scheduled_state.cancel()
            self.scheduled_state = None

            self.tracker.update_epoch(new_epoch=self.state_averager.local_epoch)
            self._should_check_synchronization_on_update = True
            # the above line ensures that peers check for *strict* synchronization once per epoch

            if not self.client_mode:
                self.state_averager.state_sharing_priority = self.local_epoch

            logger.log(self.status_loglevel, f"Transitioning to epoch {self.local_epoch}")

    def _maybe_start_optimizer_step(self) -> None:
        """If next epoch is coming soon, start the next state averaging round at the end of epoch."""
        if not self.in_update and self.ready_to_update_epoch:
            self.in_update = True
            eta_seconds = self.tracker.estimated_next_update_time - get_dht_time()
            logger.log(self.status_loglevel, f"Starting optimizer step round in {eta_seconds:.2f} sec")

    def _is_load_state_and_join_blocked(self):
        """Check if we are within the block window from averaging round."""
        if (
            self.load_state_block_window == 0
            or self.tracker.global_epoch < self.average_state_every - self.load_state_block_window
        ):
            return False
        epoch_mod_avg_interval = self.tracker.global_epoch % self.average_state_every
        return epoch_mod_avg_interval < self.load_state_block_window or epoch_mod_avg_interval >= (
            self.average_state_every - self.load_state_block_window
        )

    def _refresh_global_epoch(self):
        """Get fresh global epoch"""
        self.tracker.fetched_global_progress_this_epoch.clear()
        # This shall trigger SIGTERM for worker from kill rules in log_monitor_rules:
        if not self.tracker.fetched_global_progress_this_epoch.wait(timeout=600):
            logger.error("Timed out waiting for global progress update")

    def load_state_from_peers(self, wait_for_end_round=False, **kwargs):
        self._load_state_from_peers(**kwargs)

        self._refresh_global_epoch()
        self._resync_state()

        # FIXME: maybe move this into _resync_state ?
        if not self.client_mode:
            self.state_averager.state_sharing_priority = self.local_epoch

        logger.info("Joining run")

    def _load_state_from_peers(self, **kwargs):
        """
        Attempt to load the newest collaboration state from other peers within the same run_id.

        If successful, this will update parameters, optimizer state, local epoch and learning rate schedule in-place.
        """
        while True:
            try:
                while self._is_load_state_and_join_blocked():
                    logger.info(
                        f"Waiting before downloading state: {(self.tracker.global_epoch % self.average_state_every)}/{self.average_state_every} epochs to averaging round."
                    )
                    time.sleep(self.load_state_retry_sleep)

                with self.tracker.pause_updates():
                    _, success = self.state_averager.load_state_from_peers(
                        timeout=self.load_state_timeout,
                        load_state_sort_strategy=self.load_state_sort_strategy,
                        **kwargs,
                    )
                if not success:
                    time.sleep(self.load_state_retry_sleep)
                    logger.warning(
                        f"No available experts to load state from. Retrying in {self.load_state_retry_sleep} sec..."
                    )
                    continue
                break
            except KeyboardInterrupt:
                raise
            except BaseException as e:
                logger.error(
                    f"Failed to load state from peers: {e}, retrying ...",
                    exc_info=True,
                )
                continue

        if self.grad_buffer is not None:
            self.grad_buffer.reset_accumulated_grads_()

    def wait_for_join_window(self) -> None:
        """Wait until the current epoch is outside the block window around averaging rounds."""
        self._refresh_global_epoch()
        while self._is_load_state_and_join_blocked():
            logger.info(
                f"Waiting for safe join window: "
                f"{self.tracker.global_epoch % self.average_state_every}/{self.average_state_every} "
                f"epochs into averaging interval."
            )
            time.sleep(self.load_state_retry_sleep)
            self._refresh_global_epoch()

    def _enter_ghost_mode(self):
        self.ghost_phase = GhostPhase.PHASE1
        self.ghost_phase_start_epoch = self.tracker.global_epoch
        self.state_averager.allow_state_sharing = False

    def _transition_to_phase2(self):
        self.ghost_phase = GhostPhase.PHASE2
        self.ghost_phase_start_epoch = self.local_epoch
        self.state_averager.allow_state_sharing = False

    def _exit_ghost_mode(self):
        self.ghost_phase = GhostPhase.OFF
        self.set_averagers_allow_state_sharing()

    @property
    def ghost_phase(self) -> GhostPhase:
        return self._ghost_phase

    @ghost_phase.setter
    def ghost_phase(self, value: GhostPhase):
        self._ghost_phase = value

    @property
    def can_checkpoint(self) -> bool:
        """
        Whether this coordinator is in a state that can be checkpointed.
        """
        return self._ghost_phase == GhostPhase.OFF

    def load_state_dict(self, state_dict: dict):
        result = super().load_state_dict(state_dict)

        self._refresh_global_epoch()
        self._resync_state()

        return result
