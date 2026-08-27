# This file contains code originally from Hivemind under MIT License
# Original: Copyright 2020 Learning@home authors and collaborators
# Modified by: Pluralis Research 2026
#
# Original code: MIT License (see THIRD_PARTY_LICENSES)
# Modifications: Apache 2.0 License (see LICENSE)
#
# Licensed under the Apache License, Version 2.0 (the "License") for modifications only;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import time

from typing import Callable  # noqa: UP035

import torch

from agora_server.core.coordination.training_coordinator_base import TrainingCoordinatorBase
from agora_server.hivemind.optim.grad_scaler import GradScaler
from agora_server.hivemind.utils import get_logger


logger = get_logger(__name__)


class TrainingCoordinatorSync(TrainingCoordinatorBase):
    """Synchronous training coordinator: the swarm transitions epochs together within the step call.

    Gradients are accumulated locally in a gradient buffer (``grad_buffer_factory``) until peers collectively reach
    `target_batch_size`; the epoch transition then runs the global optimizer step and the state averaging round
    synchronously before training continues. Peers that fall behind by more than `max_allowed_stale` epochs
    re-download state from the swarm.
    """

    def _resync_state(self):
        """
        Returns:
            bool: True if state was resynchronized from peers, False otherwise.
            If True, the step will be discarded (because of stale weights).
        """
        if self._should_load_state_from_peers():
            logger.log(self.status_loglevel, "Peer is out of sync")
            self.load_state_from_peers()
            return True  # local gradients were computed with out-of-sync parameters, must start over
        elif self._catchup_epoch():
            with self.tracker.pause_updates():
                logger.log(self.status_loglevel, f"Catching up with collaboration step {self.tracker.global_epoch}")
                self.state_averager.local_epoch = self.tracker.global_epoch
                self.tracker.report_local_progress(local_epoch=self.local_epoch, samples_accumulated=0)
        return False

    def _catchup_epoch(self) -> bool:
        return (self.tracker.global_epoch - self.max_allowed_stale) <= self.local_epoch < self.tracker.global_epoch

    def _auto_step(self) -> None:
        """Internal method to call `step()` automatically."""
        try:
            # Call the parent class's step method with one batch size
            logger.debug(f"Auto step at {time.strftime('%H:%M:%S')}")
            self._last_step_time = time.time()

            if self._resync_state():
                return None

            if self.in_update and self.tracker.ready_to_update_epoch:
                batch_size = 1
                self._step(batch_size=batch_size)

            self.state_averager.allow_state_sharing = self.allow_state_sharing
        except Exception as e:
            logger.error(f"Error in auto step: {e}")

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
        self.state_averager.step(apply_delayed_updates=True)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        grads_are_valid = self._check_and_accumulate_gradients(batch_size, grad_scaler)
        if not grads_are_valid:
            return loss  # local gradients were reset due to overflow, must start over

        if self.in_update and self.tracker.ready_to_update_epoch:
            self.state_averager.allow_state_sharing = False  # Prevent state sharing during AR.
            with self.optimizer_lock:
                self._update_global_epoch(grad_scaler)

            if self.model.model_args.use_compression:
                self.model.ss_regularize()

            self.in_update = False
        return loss

    def _update_global_epoch(self, grad_scaler: GradScaler | None) -> None:
        """Depending on the configuration: aggregate gradients and/or parameters, perform global optimizer step."""
        assert self._schema_hash == self._compute_schema_hash(), "parameters or gradients changed during iteration"
        _epoch_start_time = time.perf_counter()

        with self.tracker.pause_updates():
            next_epoch = max(self.local_epoch + 1, self.tracker.global_epoch)
            self.state_averager.global_epoch = next_epoch - 1

            swarm_not_empty = self.tracker.global_progress.num_peers > 1
            should_perform_optimizer_step = not self.auxiliary and not self.use_local_updates
            should_average_state = (
                swarm_not_empty
                and next_epoch % self.average_state_every == 0
                and not self.state_averager.averaging_in_progress
            )

            if should_average_state and self.scheduled_state is not None:
                if self.scheduled_state.triggered or self.scheduled_state.done():
                    logger.log(
                        self.status_loglevel,
                        f"Not using pre-scheduled group for state averaging because it"
                        f"was already used elsewhere: {self.scheduled_state}",
                    )
                    self.scheduled_state = None
                self.delay_before_state_averaging.update(task_size=1, interval=time.perf_counter() - _epoch_start_time)

            self.state_averager.step(
                increment_epoch=True,
                wait_for_trigger=None,
                optimizer_step=should_perform_optimizer_step,
                delay_optimizer_step=self.delay_optimizer_step and should_perform_optimizer_step,
                grad_scaler=grad_scaler,
                averaging_round=should_average_state,
                delay_averaging=self.delay_state_averaging and not self.auxiliary,
                averaging_control=self.scheduled_state if should_average_state else None,
                averaging_opts=dict(timeout=self.averaging_timeout) if should_average_state else None,
            )

            if not should_average_state and self.scheduled_state is not None and not self.scheduled_state.done():
                self.scheduled_state.cancel()
            self.scheduled_state = None

            self.tracker.update_epoch(new_epoch=self.state_averager.local_epoch)

            if not self.client_mode:
                self.state_averager.state_sharing_priority = self.local_epoch

            logger.log(self.status_loglevel, f"Transitioning to epoch {self.local_epoch}")

    def _should_load_state_from_peers(self) -> bool:
        """If true, peer will discard local progress and attempt to download state from peers."""
        return self.local_epoch < (self.tracker.global_epoch - self.max_allowed_stale)

    def load_state_from_peers(self, wait_for_end_round=False, **kwargs):
        # Wait for a while grad accumulation round before requesting state from peers
        logger.info("Waiting for peers in stage to finish step before joining")
        while True:
            if (
                self.tracker.fetched_global_progress_this_epoch.is_set()
                and self.tracker.global_progress.samples_accumulated < self.target_batch_size * 0.1
            ):
                break
            else:
                logger.info("Waiting for peers in stage to finish step before joining")
                time.sleep(self.tracker.max_refresh_period)

        self._load_state_from_peers(**kwargs)

        if wait_for_end_round:
            self.tracker.fetched_global_progress_this_epoch.clear()
            while True:
                if (
                    self.tracker.fetched_global_progress_this_epoch.is_set()
                    and self.tracker.global_progress.samples_accumulated < self.target_batch_size * 0.2
                ):
                    break
                else:
                    logger.info("Downloaded state, waiting for start of new round")
                    time.sleep(0.5)

        logger.info("Joining run")

    def _load_state_from_peers(self, **kwargs):
        """
        Attempt to load the newest collaboration state from other peers within the same run_id.

        If successful, this will update parameters, optimizer state, local epoch and learning rate schedule in-place.
        """
        with self.tracker.pause_updates():
            while True:
                try:
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

            if self.tracker.global_epoch - 1 <= self.local_epoch < self.tracker.global_epoch:
                logger.log(self.status_loglevel, f"Catching up with collaboration step {self.tracker.global_epoch}")
                self.state_averager.local_epoch = self.tracker.global_epoch

            self.tracker.report_local_progress(local_epoch=self.local_epoch, samples_accumulated=0)

            if not self.client_mode:
                self.state_averager.state_sharing_priority = self.local_epoch
