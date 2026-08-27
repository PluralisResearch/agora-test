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

import logging
import multiprocessing as mp
import os
import threading
import time

from abc import abstractmethod
from multiprocessing.synchronize import Lock
from typing import Any, Callable, Literal, Sequence  # noqa: UP035

import torch

from agora_server.core.averaging.averager import PeerSortStrategy
from agora_server.core.averaging.gradient_buffer import GradientBuffer, GradientBufferFactory
from agora_server.core.averaging.state_averager import TrainingStateAverager, TrainingStateAveragerFactory
from agora_server.core.coordination.progress_tracker import LocalTrainingProgress, ProgressTracker
from agora_server.core.coordination.training_coordinator import TrainingCoordinator
from agora_server.hivemind.averaging.control import StepControl
from agora_server.hivemind.dht import DHT
from agora_server.hivemind.optim.grad_scaler import GradScaler
from agora_server.hivemind.utils import PerformanceEMA, get_logger
from agora_server.types import (
    LRSchedulerBase,
    Parameters,
    ParamGroups,
    SchedulerFactory,
    TorchOptimizer,
    TorchOptimizerFactory,
)


logger = get_logger(__name__)


class TrainingCoordinatorBase(TrainingCoordinator):
    """Shared chassis for training coordinators built around a state averager and a progress tracker.

    This class owns everything a training coordinator needs regardless of its update policy: construction of the
    ProgressTracker, TrainingStateAverager and optional GradientBuffer, the auto-step monitor thread, gradient
    accumulation and validity checks, checkpoint (de)serialization, and the torch-optimizer facade (``opt``,
    ``param_groups``, ``state``, ``zero_grad``).

    Subclasses implement the update policy — when and how the swarm transitions epochs — through the abstract hooks
    declared below (``_step``, ``_auto_step``, ``_update_global_epoch``, ``_resync_state``, ``_catchup_epoch``,
    ``_should_load_state_from_peers``, ``_load_state_from_peers``) plus ``load_state_from_peers``, which stays
    abstract from TrainingCoordinator. The public ``step`` acquires the step lock, re-synchronizes with the swarm and
    delegates to ``_step``; the monitor thread calls ``_auto_step`` whenever no external step arrived within
    ``auto_step_time``.
    """

    def __init__(
        self,
        *,
        dht: DHT,
        run_id: str,
        target_batch_size: int,
        model: Any,  # TODO: change to BaseExpert
        optimizer_lock: Lock,
        state_avg_factory: TrainingStateAveragerFactory,
        auto_step_time: float = 3.0,
        max_allowed_stale: int = 3,
        load_state_retry_sleep: float = 5.0,
        allow_state_sharing: bool = True,
        averaging_mode: Literal["client", "node"] = "node",
        load_state_sort_strategy: PeerSortStrategy = "uid",
        batch_size_per_step: int | None = None,
        optimizer: TorchOptimizer | TorchOptimizerFactory,
        params: Parameters | ParamGroups | None = None,
        scheduler: LRSchedulerBase | SchedulerFactory | None = None,
        matchmaking_time: float = 15.0,
        averaging_timeout: float | None = 60.0,
        allreduce_timeout: float | None = 60.0,
        next_chunk_timeout: float = 10.0,
        load_state_timeout: float = 600.0,
        reuse_grad_buffers: bool = False,
        offload_optimizer: bool = False,
        delay_optimizer_step: bool = False,
        delay_grad_averaging: bool = False,
        delay_state_averaging: bool = False,
        average_state_every: int = 1,
        use_local_updates: bool = False,
        client_mode: bool | None = None,
        auxiliary: bool = False,
        bandwidth: float | None = None,
        grad_buffer_factory: GradientBufferFactory | None = None,
        average_opt_statistics: Sequence[str] = (),
        extra_tensors: Sequence[torch.Tensor] = (),
        tracker_opts: dict | None = None,
        performance_ema_alpha: float = 0.1,
        shutdown_timeout: float = 5,
        verbose: bool = False,
    ):
        """Initialize the training coordinator with distributed training capabilities.

        Args:
            dht (DHT): A running hivemind.DHT instance connected to other peers.
            run_id (str): A unique identifier of this training run, used as a common prefix for all DHT keys. Peers with the same run_id should *generally* train the same model and use compatible configurations. Some options can be safely changed by individual peers: ``batch_size_per_step``, ``client_mode``, ``auxiliary``, ``reuse_grad_buffers``, ``offload_optimizer``, and ``verbose``. In some cases, other options may also be tuned individually by each peer, but they should be changed with caution to avoid deadlocks or convergence issues.
            target_batch_size (int): Global batch size that must be accumulated before the swarm transitions to the next epoch. The actual batch may be *slightly* larger due asynchrony (e.g. peers submit more gradients in the last second).
            model (Any): Model being trained.
            optimizer_lock (Lock): A lock to protect optimizer state across multiple processes.
            state_avg_factory (TrainingStateAveragerFactory): A callable that creates TrainingStateAverager with required averaging strategy.
            auto_step_time (float, optional): If optimizer step is not performed within this time (in seconds), do an auto step. Defaults to 3.0.
            max_allowed_stale (int, optional): Maximum number of stale epochs before a forced synchronization. Defaults to 3.
            load_state_retry_sleep (float, optional): Time (in seconds) to wait before retrying load_state_from_peers after a failure. Defaults to 5.0.
            allow_state_sharing (bool, optional): If False, peer will not share its state with others during load_state_from_peers. Defaults to True.
            averaging_mode (Literal["client", "node"], optional): Whether to use client-mode or node-mode during averaging. Defaults to "node".
            load_state_sort_strategy (PeerSortStrategy, optional): Strategy to sort peers when loading state from multiple peers. Defaults to "uid".
            batch_size_per_step (int | None, optional): You should accumulate gradients over this many samples between calls to optimizer.step. Defaults to None.
            optimizer (TorchOptimizer | OptimizerFactory): A callable(parameters) -> pytorch.optim.Optimizer or a pre-initialized PyTorch optimizer. Some advanced options like offload_optimizer, delay_optimizer_step, or delay_grad_averaging require and require the callable and will not work if hivemind.optimizer is created with a pre-existing PyTorch Optimizer.
            params (Parameters | ParamGroups | None, optional): Parameters or param groups for the optimizer; required if optimizer is a callable(params). Defaults to None.
            scheduler (LRSchedulerBase | SchedulerFactory | None, optional): Callable(optimizer) -> PyTorch LRScheduler or a pre-initialized PyTorch scheduler. The learning rate scheduler will adjust learning rate based on global epoch, not the number of local calls to optimizer.step; this is required to keep different peers synchronized. Defaults to None.
            matchmaking_time (float, optional): When looking for group, wait for peers to join for up to this many seconds. Increase if you see "averaged gradients with N peers" where N is below 0.9x the real size on >=25% of epochs. When training with low-latency network, decreasing matchmaking_time allows training with smaller batch sizes. Defaults to 15.0.
            averaging_timeout (float | None, optional): If an averaging step hangs for this long, it will be cancelled automatically. Increase averaging_timeout if you see "Proceeding with local gradients" at least 25% of the time. Do not set this timeout too high, as it may cause your optimizer to hang after some types of network errors. Defaults to 60.0.
            allreduce_timeout (float | None, optional): Timeout for a single attempt to run all-reduce. Defaults to 60.
            next_chunk_timeout (float, optional): Timeout for receiving next chunk during all-reduce. Defaults to 10.
            load_state_timeout (float, optional): Wait for at most this many seconds before giving up on load_state_from_peers. Defaults to 600.0.
            reuse_grad_buffers (bool, optional): If True, use model's .grad buffers for gradient accumulation. This is more memory efficient, but it requires that the user does *NOT* call model/opt zero_grad at all. Defaults to False.
            offload_optimizer (bool, optional): Offload the optimizer to host memory, saving GPU memory for parameters and gradients. Defaults to False.
            delay_optimizer_step (bool, optional): Run optimizer in background, apply results in future .step; requires offload_optimizer. Defaults to False.
            delay_grad_averaging (bool, optional): Average gradients in background; requires offload_optimizer and delay_optimizer_step. Defaults to False.
            delay_state_averaging (bool, optional): If enabled (default), average parameters and extra tensors in a background thread; if set to False, average parameters synchronously within the corresponding hivemind.Optimizer.step call. Defaults to True.
            average_state_every (int, optional): Average state (parameters, chosen opt tensors) with peers every this many **epochs**. This reduces the communication overhead increasing, but can cause parameters to diverge if too large. The maximal average_state_every=num_epochs depends on how often peers diverge from each other. If peers hardly ever skip averaging rounds, they can average state less frequently. In turn, network failures, lossy gradient compression and local_updates cause parameters to diverge faster and requires more frequent averaging. Defaults to 1.
            use_local_updates (bool, optional): If enabled, peers will update parameters on each .step using local gradients; if not enabled (default), accumulate gradients to target_batch_size, and then call .step with averaged gradients. Even if use_local_updates=True, learning rate scheduler will still be called once per target_batch_size. Defaults to False.
            client_mode (bool | None, optional): If True, this peer will not accept incoming connections (firewall-compatible mode). Defaults to None.
            auxiliary (bool, optional): If True, optimizer.step will only assist other peers in averaging (for cpu-only workers). Defaults to False.
            bandwidth (float | None, optional): If specified, this value represents the network bandwidth available to averager. By default, the averager is assumed to have the average bandwidth of his group. If bandwidth == 0, averager will rely on its groupmates to do all the averaging. Defaults to None.
            average_opt_statistics (Sequence[str], optional): Names of optimizer statistics from state dict that should be averaged with peers. Defaults to ().
            extra_tensors (Sequence[torch.Tensor], optional): If specified, these extra tensors will also be averaged and shared in load_state_from_peers. Defaults to ().
            tracker_opts (dict | None, optional): Additional keyword arguments forwarded to ProgressTracker. Defaults to None.
            performance_ema_alpha (float, optional): Moving average alpha in ProgressTracker, TrainingStateAverager and Optimizer. Defaults to 0.1.
            shutdown_timeout (float, optional): Maximum time to wait for graceful shutdown. Defaults to 5.
            verbose (bool, optional): If True, report internal events such as accumulating gradients and running background tasks. Defaults to False.

        Note:
            In a large-scale training, peers will inevitably fail and you will see error messages. The training coordinator is designed to recover from such failures, but will sometimes need a minute or two to re-adjust.
        """
        self._parent_pid = os.getpid()

        client_mode = client_mode if client_mode is None else dht.client_mode
        assert not delay_grad_averaging or delay_optimizer_step, "delay_grad_averaging requires delay_optimizer_step"
        assert not (client_mode and auxiliary), "Client-mode peers cannot serve as auxiliaries"
        assert not auxiliary or batch_size_per_step is None, "Auxiliary peers should not accumulate batches"
        if callable(optimizer) and params is not None:
            if scheduler is not None and (not callable(scheduler) or isinstance(scheduler, LRSchedulerBase)):
                raise ValueError("For this mode, please provide scheduler factory: callable(optimizer) -> scheduler")
        elif all(hasattr(optimizer, attr) for attr in ("param_groups", "step", "zero_grad")):
            if offload_optimizer or delay_optimizer_step or delay_grad_averaging:
                raise ValueError(
                    "To enable offload_optimizer or delayed updates, please initialize Optimizer as "
                    "hivemind.Optimizer(..., params=params, optimizer=lambda params: create_opt(params)"
                )
        else:
            raise ValueError(
                "Please initialize the optimizer in one of the following two ways:\n"
                "(A) hivemind.Optimizer(..., params=params, optimizer=lambda params: create_opt(params)\n"
                "(B) hivemind.Optimizer(..., optimizer=pre_initialize_optimizer)"
            )
        if use_local_updates:
            assert not reuse_grad_buffers, "if local_updates is True, gradients will not be accumulated"
            assert not delay_grad_averaging, "if local_updates is True, gradients will not be averaged"

        self._auto_step_time = auto_step_time
        self.max_allowed_stale = max_allowed_stale
        self.optimizer_lock = optimizer_lock
        self.load_state_retry_sleep = load_state_retry_sleep
        self.allow_state_sharing = allow_state_sharing
        self.averaging_mode = averaging_mode
        self.load_state_sort_strategy = load_state_sort_strategy
        self.model = model

        self.dht, self.run_id, self.client_mode, self.auxiliary = dht, run_id, client_mode, auxiliary
        self.batch_size_per_step, self.target_batch_size = batch_size_per_step, target_batch_size
        self.delay_state_averaging, self.average_state_every = delay_state_averaging, average_state_every
        self.matchmaking_time, self.offload_optimizer = matchmaking_time, offload_optimizer
        self.delay_grad_averaging, self.delay_optimizer_step = delay_grad_averaging, delay_optimizer_step

        self.averaging_timeout, self.allreduce_timeout = averaging_timeout, allreduce_timeout
        self.load_state_timeout, self.shutdown_timeout = load_state_timeout, shutdown_timeout
        self.next_chunk_timeout = next_chunk_timeout

        self.status_loglevel = logging.INFO if verbose else logging.DEBUG
        self.scheduled_state: StepControl | None = None

        self._last_step_time: float = time.time()
        self._step_lock = mp.Lock()
        self._monitor_thread: threading.Thread | None = None
        self._should_stop = threading.Event()
        self.in_update = False

        # Initialize Progress Tracker
        self.tracker = self._make_progress_tracker(
            target_batch_size, performance_ema_alpha=performance_ema_alpha, **tracker_opts or {}
        )

        # Initialize State Averager
        self.state_averager = self._make_state_averager(
            state_avg_factory,
            optimizer=optimizer,
            params=params,
            scheduler=scheduler,
            average_opt_statistics=average_opt_statistics,
            performance_ema_alpha=performance_ema_alpha,
            extra_tensors=extra_tensors,
            average_state_every=self.average_state_every,
            bandwidth=bandwidth,
        )

        # Initialize the local gradient buffer.
        self.grad_averager = None
        self.grad_buffer = None
        if auxiliary:
            # An auxiliary peer never computes gradients, it only aggregates other peers' parts.
            logger.log(self.status_loglevel, "Auxiliary peer: no gradient source")
        elif grad_buffer_factory is not None:
            assert not use_local_updates, "grad_buffer_factory is incompatible with use_local_updates"
            self.grad_buffer = self._make_gradient_buffer(grad_buffer_factory, reuse_grad_buffers=reuse_grad_buffers)
        elif not use_local_updates:
            raise ValueError(
                "Non-auxiliary peers must provide grad_buffer_factory unless use_local_updates is enabled"
            )

        self._schema_hash = self._compute_schema_hash()

        self.delay_before_state_averaging = PerformanceEMA(alpha=performance_ema_alpha)
        # measures the average time from the beginning of self._update_global_epoch to the call to state_averager
        # used for pre-scheduling the averaging round in state_averager

        self._step_supports_amp_scaling = reuse_grad_buffers
        # note: the line above is used by pytorch AMP GradScaler to enable custom behavior needed when reusing gradient
        # buffers over multiple steps (to avoid repeated unscaling). Without reuse_grad_buffers, this is not needed.

    def _make_state_averager(self, factory: TrainingStateAveragerFactory, **kwargs) -> TrainingStateAverager:
        return factory(
            dht=self.dht,
            prefix=f"{self.run_id}_state_averager",
            min_matchmaking_time=self.matchmaking_time,
            allreduce_timeout=self.allreduce_timeout,
            shutdown_timeout=self.shutdown_timeout,
            offload_optimizer=self.offload_optimizer,
            custom_gradients=self.offload_optimizer,
            status_loglevel=self.status_loglevel,
            next_chunk_timeout=self.next_chunk_timeout,
            client_mode=self.client_mode,
            auxiliary=self.auxiliary,
            allow_state_sharing=False,  # Always forbid state sharing at init; will be updated later
            mode=self.averaging_mode,
            start=True,
            **kwargs,
        )

    def _make_gradient_buffer(self, factory: GradientBufferFactory, **kwargs) -> GradientBuffer:
        assert hasattr(self, "state_averager"), "must initialize state averager first"
        grad_buffer = factory(parameters=self.state_averager.main_parameters, **kwargs)
        assert isinstance(grad_buffer, GradientBuffer)
        self._alias_offloaded_grads(grad_buffer)
        return grad_buffer

    def _alias_offloaded_grads(self, grad_buffer: GradientBuffer) -> None:
        """If the optimizer is offloaded, point its parameters .grad at the gradient source host
        (CPU) buffers, so loading accumulated gradients populates the offloaded optimizer directly."""
        if not self.offload_optimizer:
            return
        optimized_parameters = [
            param for group in self.state_averager.optimizer.param_groups for param in group["params"]
        ]
        with grad_buffer.get_averaged_grads() as averaged_gradients:
            assert len(averaged_gradients) == len(optimized_parameters)
            for opt_param, averaged_grad in zip(optimized_parameters, averaged_gradients):
                opt_param.grad = averaged_grad

    def _make_progress_tracker(self, target_batch_size: int, **kwargs) -> ProgressTracker:
        return ProgressTracker(
            dht=self.dht,
            prefix=self.run_id,
            target_batch_size=target_batch_size,
            client_mode=self.client_mode,
            status_loglevel=self.status_loglevel,
            start=True,
            **kwargs,
        )

    def _compute_schema_hash(self) -> int:
        optimized_param_groups = self.state_averager.optimizer.param_groups
        optimized_parameters = [param for group in optimized_param_groups for param in group["params"]]
        param_shapes = tuple(tuple(param.shape) for param in optimized_parameters)

        # offloaded optimizer requires that gradient tensors are reused between iterations
        grad_ids = tuple(id(param.grad) for param in optimized_parameters) if self.offload_optimizer else None
        return hash((grad_ids, param_shapes))

    @abstractmethod
    def _step(
        self,
        closure: Callable[[], torch.Tensor] | None = None,
        batch_size: int | None = None,
        grad_scaler: GradScaler | None = None,
    ) -> torch.Tensor | None:
        """Run one collaborative step: report progress, and transition the epoch when the swarm is ready.

        Returns:
            torch.Tensor | None: The loss returned by ``closure``, if one was given.
        """

    @abstractmethod
    def _auto_step(self) -> None:
        """Step invoked by the monitor thread when no external step arrived within ``auto_step_time``."""

    @abstractmethod
    def _update_global_epoch(self, grad_scaler: GradScaler | None) -> None:
        """Transition to the next epoch: apply the optimizer update and/or state averaging per the update policy."""

    @abstractmethod
    def _resync_state(self) -> bool:
        """Re-synchronize with the swarm before stepping.

        Returns:
            bool: True if state was re-downloaded from peers and the current step must be discarded.
        """

    @abstractmethod
    def _catchup_epoch(self) -> bool:
        """Whether ``local_epoch`` should be fast-forwarded to the global epoch."""

    @abstractmethod
    def _should_load_state_from_peers(self) -> bool:
        """If true, peer will discard local progress and attempt to download state from peers."""

    @abstractmethod
    def _load_state_from_peers(self, **kwargs):
        """Download and apply the newest collaboration state from peers, retrying until success."""

    def is_alive(self) -> bool:
        return self.state_averager.is_alive()

    @property
    def local_epoch(self) -> int:
        """This worker's current epoch, kept synchronized with peers.

        If peer's local_epoch lags behind others, it will automatically re-synchronize by downloading state from another peer.
        An epoch corresponds to accumulating target_batch_size across all active devices.
        """
        return self.state_averager.local_epoch

    @property
    def local_progress(self) -> LocalTrainingProgress:
        return self.tracker.local_progress

    @property
    def use_local_updates(self) -> bool:
        return self.grad_buffer is None

    @property
    def ready_to_update_epoch(self) -> bool:
        """Whether or not this peer can increment epoch right away."""
        return (
            self.tracker.global_epoch > self.tracker.local_progress.epoch
            or self.tracker.global_progress.samples_accumulated >= self.tracker.target_batch_size
        )

    def start_monitor(self) -> None:
        """Start the monitoring thread if it's not already running."""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._should_stop.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_step, daemon=True)
            self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        """Stop the monitoring thread."""
        self._should_stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=1)
            self._monitor_thread = None

    def _monitor_step(self) -> None:
        """Monitor thread that checks and calls `step()` if it wasn't called within auto_step_time window."""
        while not self._should_stop.is_set():
            time.sleep(1.0)  # Check every 1sec

            current_time = time.time()
            time_since_last_step = current_time - self._last_step_time

            if time_since_last_step >= self._auto_step_time:
                with self._step_lock:
                    # Check again after acquiring lock in case step() was called
                    if time.time() - self._last_step_time >= self._auto_step_time:
                        self._auto_step()

    def step(self, batch_size: int | None = None) -> None:
        """Override of the step method that updates the last step time.

        This should be called by external code.
        """
        with self._step_lock:
            if self._resync_state():
                return None

            self._step(batch_size=batch_size)
            self.state_averager.allow_state_sharing = self.allow_state_sharing

            self._last_step_time = time.time()

    def _check_and_accumulate_gradients(
        self, batch_size: int, grad_scaler: GradScaler | None = None, override_samples_accumulated: int | None = None
    ) -> bool:
        """Check if gradients are valid, accumulate and return True; otherwise, reset and return False or exit run.

        Args:
            batch_size: Number of samples in this batch.
            grad_scaler: Optional gradient scaler.
            override_samples_accumulated: If set, report this value to the progress tracker instead of the
                actual accumulated samples. Used in ghost phase 2 to report 0 samples while still accumulating
                real gradients for optimizer warm-up.
        """
        if self.grad_buffer.has_nan_grads():
            self.tracker.report_local_progress(self.local_epoch, samples_accumulated=0)
            logger.error("Encountered incorrect value in grads, resetting local gradients")
            self.grad_buffer.reset_accumulated_grads_()
            return False

        self.grad_buffer.accumulate_grads_(batch_size)
        reported_samples = (
            override_samples_accumulated
            if override_samples_accumulated is not None
            else self.grad_buffer.local_samples_accumulated
        )
        self.tracker.report_local_progress(self.local_epoch, reported_samples)
        return True

    def _load_averaged_gradients_into_optimizer_(self):
        """If required, load averaged gradients into optimizer."""
        assert self.grad_buffer is not None

        if self.offload_optimizer:
            pass  # averaged gradients are already baked into optimizer, see _make_gradient_buffer
        else:
            # copy averaged gradients into optimizer .grad buffers
            optimized_param_groups = self.state_averager.optimizer.param_groups
            optimized_parameters = [param for group in optimized_param_groups for param in group["params"]]
            with torch.no_grad(), self.grad_buffer.get_averaged_grads() as averaged_gradients:
                assert len(averaged_gradients) == len(optimized_parameters)
                for opt_param, averaged_grad in zip(optimized_parameters, averaged_gradients):
                    if opt_param.grad is None:
                        opt_param.grad = averaged_grad.clone()
                    else:
                        opt_param.grad.copy_(averaged_grad, non_blocking=True)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Reset gradients from model. If reuse_grad_buffers=True, this will raise an error."""
        if self.grad_buffer is not None and self.grad_buffer.reuse_grad_buffers:
            raise ValueError(
                f"When running {self.__class__.__name__} with reuse_grad_buffers=True, user should never "
                f"call zero_grad manually. Gradients will be refreshed internally"
            )
        for param_group in self.param_groups:
            for param in param_group["params"]:
                if set_to_none:
                    param.grad = None
                elif param.grad is not None:
                    param.grad.zero_()

    def should_zero_grad_after_backward(self) -> bool:
        """Whether ModuleCollab should clear model gradients after a batch backward pass."""
        return not (self.grad_buffer is not None and self.grad_buffer.reuse_grad_buffers)

    def set_averagers_allow_state_sharing(self) -> None:
        self.state_averager.allow_state_sharing = self.allow_state_sharing

    def state_dict(self) -> dict:
        state_dict = self.state_averager.optimizer.state_dict()
        state_dict["state"]["local_epoch"] = self.local_epoch
        return state_dict

    def load_state_dict(self, state_dict: dict):
        if "local_epoch" in state_dict["state"]:
            self.state_averager.local_epoch = state_dict["state"].pop("local_epoch")
        state_dict["state"].pop("grad_averager_buffers", None)
        state_dict["state"].pop("error_buffers", None)
        result = self.state_averager.optimizer.load_state_dict(state_dict)

        # After checkpoint load, sync GPU main_parameters -> CPU optimizer params.
        if self.state_averager.offload_optimizer:
            self.state_averager._load_main_params_into_optimizer_()
        if not self.state_averager.reuse_tensors:
            self.state_averager._load_local_tensors_into_averager_()
        self.state_averager._update_scheduler()

        return result

    @property
    def state(self):
        return dict(self.state_averager.optimizer.state, local_epoch=self.local_epoch)

    @property
    def opt(self) -> TorchOptimizer:
        return self.state_averager.optimizer

    @property
    def param_groups(self) -> ParamGroups:
        next_index = 0
        param_groups = tuple(dict(param_group) for param_group in self.state_averager.optimizer.param_groups)
        for param_group in param_groups:
            num_params = len(param_group["params"])
            main_params_for_group = self.state_averager.main_parameters[next_index : next_index + num_params]
            param_group["params"] = main_params_for_group
            next_index += num_params
        assert next_index == len(self.state_averager.main_parameters)
        return param_groups

    def __repr__(self):
        return f"{self.__class__.__name__}(prefix={self.run_id}, epoch={self.local_epoch})"

    def shutdown(self):
        logger.log(self.status_loglevel, "Sending goodbye to peers...")
        self.stop_monitoring()
        self.tracker.shutdown(self.shutdown_timeout)

        # Cancel any pending delayed updates
        for pending_update in self.state_averager.pending_updates:
            if not pending_update.done():
                pending_update.cancel()
        self.state_averager.pending_updates.clear()

        # Cancel the scheduled averaging round
        if self.scheduled_state is not None and not self.scheduled_state.done():
            self.scheduled_state.cancel()

        logger.log(self.status_loglevel, "Shutting down averagers...")
        self.state_averager.shutdown()
        logger.log(self.status_loglevel, f"{self.__class__.__name__} is shut down")

    def __del__(self):
        if self._parent_pid == os.getpid() and self.is_alive():
            self.shutdown()
