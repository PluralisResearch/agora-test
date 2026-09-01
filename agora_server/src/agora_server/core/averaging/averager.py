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

"""A background process that averages your tensors with peers"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import functools
import gc
import multiprocessing as mp
import os
import random
import signal
import threading
import time
import weakref

from dataclasses import asdict
from typing import (  # noqa: UP035 p2p download doesn't work when AsyncIterator is imported from collections.abc
    Any,
    AsyncIterator,
    Literal,
    Sequence,
)

import numpy as np
import torch

import agora_server.core.averaging.compression  # noqa: F401 - registers BF16 compression handler

from agora_server.core.all_reduce.ar_runner import AllreduceException, AllReduceRunner, AveragingMode, GroupID
from agora_server.core.all_reduce.matchmaking import Matchmaking, MatchmakingException
from agora_server.core.all_reduce.partition import DEFAULT_PART_SIZE_BYTES
from agora_server.hivemind.averaging.control import AveragingStage, StepControl
from agora_server.hivemind.averaging.group_info import GroupInfo
from agora_server.hivemind.averaging.load_balancing import load_balance_peers
from agora_server.hivemind.compression import CompressionBase, CompressionInfo, NoCompression, deserialize_torch_tensor
from agora_server.hivemind.dht import DHT, DHTID
from agora_server.hivemind.p2p import P2P, P2PContext, P2PDaemonError, P2PHandlerError, PeerID, ServicerBase
from agora_server.hivemind.proto import averaging_pb2
from agora_server.hivemind.utils import MPFuture, TensorDescriptor, get_logger
from agora_server.hivemind.utils.asyncio import (
    achain,
    aiter_with_timeout,
    anext,
    as_aiter,
    azip,
    enter_asynchronously,
    switch_to_uvloop,
)
from agora_server.hivemind.utils.serializer import MSGPackSerializer, SerializerBase
from agora_server.hivemind.utils.streaming import combine_from_streaming, split_for_streaming
from agora_server.hivemind.utils.timed_storage import DHTExpiration, ValueWithExpiration, get_dht_time
from agora_server.utils.common import build_cls


# flavour types
GatheredData = Any
PeerSortStrategy = Literal["priority", "uid"]

logger = get_logger(__name__)

@functools.lru_cache(maxsize=1)
def _libc_malloc_trim():
    """Resolve glibc's malloc_trim, or None where unavailable. Only auxiliary averagers call this."""
    try:
        return ctypes.CDLL("libc.so.6").malloc_trim
    except (OSError, AttributeError):
        return None


class DecentralizedAverager(mp.Process, ServicerBase):
    """Parameter averaging service. A trainer can run this service in background to periodically average his parameters
    with other trainers. The averaging pattern is chosen so that (1) you only need to average with a small
    group of peers at a time, but (2) all trainers will converge to global average in a logarithmic number of steps.

    Example:

    >>> averager = DecentralizedAverager(...)
    >>> with averager.get_tensors() as tensors:
    >>>     # run some code, modify tensors if necessary
    >>>     tensors[0] += 1
    >>> # do not use tensors after the lock is released
    >>> metadata = averager.step(gather=dict(my_batch_size=32))
    >>> # run averaging once (in-place), gather metadata from groupmates
    >>> with averager.get_tensors() as tensors_after_averaging:
    >>>     pass # use the averaged tensors
    """

    _matchmaking: Matchmaking
    _pending_groups_registered: asyncio.Event
    _state_updated: asyncio.Event
    _p2p: P2P
    serializer = MSGPackSerializer

    def __init__(
        self,
        averaged_tensors: Sequence[torch.Tensor],
        dht: DHT,
        *,
        start: bool,
        prefix: str,
        initial_group_bits: str = "",
        min_matchmaking_time: float = 5.0,
        matchmaking_class: str = "agora_server.core.all_reduce.matchmaking.Matchmaking",
        ar_runner_class: str = "agora_server.core.all_reduce.ar_runner.AllReduceRunner",
        request_timeout: float = 3.0,
        averaging_alpha: float = 1.0,
        part_size_bytes: int = DEFAULT_PART_SIZE_BYTES,
        allreduce_timeout: float | None = None,
        next_chunk_timeout: float | None = None,
        sender_timeout: float | None = None,
        reducer_timeout: float | None = None,
        compression: CompressionBase = NoCompression(),
        state_compression: CompressionBase = NoCompression(),
        tensor_infos: Sequence[CompressionInfo] | None = None,
        bandwidth: float | None = None,
        min_vector_size: int = 0,
        auxiliary: bool = False,
        allow_state_sharing: bool | None = None,
        declare_state_period: float = 30,
        client_mode: bool | None = None,
        mode: Literal["client", "node"] = "node",
        daemon: bool = True,
        shutdown_timeout: float = 5,
        **matchmaking_kwargs,
    ):
        """Initialize DecentralizedAverager process.

        Args:
            averaged_tensors (Sequence[torch.Tensor]): A sequence of pytorch tensors that will be averaged in each all-reduce.
            dht (DHT): A DHT node that will be used to find groups.
            start (bool): If True, starts the background process immediately.
            prefix (str): A shared prefix for all group keys.
            initial_group_bits (str, optional): A string of bits ('0' and '1') that define the initial group key (bucket index). Defaults to "".
            min_matchmaking_time (float, optional): When looking for group, wait for requests for at least this many seconds. Defaults to 5.0.
            ar_runner_class (str, optional): Full dotted path to the AllReduceRunner class to use. Defaults to "agora_server.core.all_reduce.ar_runner.AllReduceRunner".
            matchmaking_class (str, optional): The class to use for matchmaking. Defaults to "agora_server.core.all_reduce.matchmaking.Matchmaking".
            request_timeout (float, optional): When looking for group, wait for a response from leader for at most this many seconds. **NOTE**: `request_timeout` must be smaller than `min_matchmaking_time` to avoid potential deadlocks. Defaults to 3.0.
            averaging_alpha (float, optional): Optional "learning rate" for averaging. If specified, local parameters will be shifted towards the (estimated) average by this coefficient. By default, local parameters are set equal to average. Defaults to 1.0.
            part_size_bytes (int, optional): Tensors for AllReduce are processed in parts of up to this size (after compression). Defaults to DEFAULT_PART_SIZE_BYTES.
            allreduce_timeout (float | None, optional): Spend at most this many seconds for all-reduce (after group is formed). Defaults to None.
            next_chunk_timeout (float | None, optional): During all-reduce and `load_state_from_peers`, if peer does not send next data chunk in this number of seconds, consider it failed and proceed with remaining peers. Defaults to None.
            sender_timeout (float | None, optional): During all-reduce, any sender that fails to send tensor chunk within this many seconds from previous chunk will be marked as failed and excluded from averaging. default: equal to `next_chunk_timeout`. Defaults to None.
            reducer_timeout (float | None, optional): During all-reduce, any reducer that fails to send results chunk within this many seconds from previous chunk will be marked as failed and excluded from averaging. default: 2 * `sender_timeout`. Defaults to None.
            compression (CompressionBase, optional): Optionally compress tensors with this compression algorithm before running all-reduce. Defaults to NoCompression().
            state_compression (CompressionBase, optional): A separate compression strategy for `load_state_from_peers`. Defaults to NoCompression().
            tensor_infos (Sequence[CompressionInfo] | None, optional): CompressionInfo for each respective tensor; this determines how the tensor will be compressed. Defaults to None.
            bandwidth (float | None, optional): If specified, this value represents the network bandwidth available to averager.By default, the averager is assumed to have the average bandwidth of his group. If bandwidth == 0, averager will rely on its groupmates to do all the averaging. Defaults to None.
            min_vector_size (int, optional): Minimum vector size for all-reduce. Defaults to 0.
            auxiliary (bool, optional): If True, averager.step will only assist others without sending local tensors for averaging. Defaults to False.
            allow_state_sharing (bool | None, optional): If set to True, other peers can download this peer's state. Can be overwritten with `averager.allow_state_sharing = True / False`. Defaults to None.
            declare_state_period (float, optional): Redeclare averager as a donor for `load_state_from_peers` every this many seconds. Defaults to 30.
            client_mode (bool | None, optional): If False, this averager will accept incoming requests from other peers. if True, the averager will only join existing groups where at least one peer has client_mode=False. By default, this flag is copied from DHTNode inside the `dht` instance. Defaults to None.
            mode (Literal["client";, "node"], optional): Averaging mode. Defaults to "node".
            daemon (bool, optional): Launch as daemon. Defaults to True.
            shutdown_timeout (float, optional): When calling .shutdown, wait for up to this many seconds before terminating. Defaults to 5.
            **matchmaking_kwargs: Additional keyword arguments forwarded to the matchmaking class constructor.
        """
        assert "." not in prefix, "group prefix must be a string without trailing '.'"
        assert bandwidth is None or (bandwidth >= 0 and np.isfinite(np.float32(bandwidth))), (
            "bandwidth must be a non-negative float32"
        )
        assert all(bit in "01" for bit in initial_group_bits)
        assert not client_mode or not auxiliary, "auxiliary peers must accept incoming connections"

        super().__init__()
        self.dht = dht
        self.prefix = prefix

        if client_mode is None:
            client_mode = dht.client_mode

        self.client_mode = client_mode

        self._parent_pid = os.getpid()

        if mode == "client" or self.client_mode:
            self.mode = AveragingMode.CLIENT
        elif auxiliary:
            self.mode = AveragingMode.AUX
        else:
            self.mode = AveragingMode.NODE

        self.daemon = daemon

        self._averaged_tensors = tuple(averaged_tensors)
        self.lock_averaged_tensors = mp.Lock()
        for tensor in self._averaged_tensors:
            assert tensor.grad_fn is None, "averaged_tensors must be either parameters or leaf tensors"
            tensor.share_memory_()
        self.total_size = sum(map(torch.Tensor.numel, self._averaged_tensors))
        self.schema_hash = compute_schema_hash(self._averaged_tensors)
        self.shutdown_timeout = shutdown_timeout
        self.next_chunk_timeout = next_chunk_timeout
        self.bandwidth = bandwidth
        self._state_download_lock = asyncio.Lock()

        self.matchmaking_class = matchmaking_class
        self._ar_runner_class = ar_runner_class

        self.matchmaking_kwargs = dict(
            request_timeout=request_timeout,
            min_matchmaking_time=min_matchmaking_time,
            **matchmaking_kwargs,
        )

        self.allreduce_kwargs = dict(
            compression=compression,
            part_size_bytes=part_size_bytes,
            min_vector_size=min_vector_size,
            sender_timeout=sender_timeout,
            reducer_timeout=reducer_timeout,
        )
        self._averaging_alpha, self._allreduce_timeout = averaging_alpha, allreduce_timeout
        self._running_groups: dict[GroupID, asyncio.Future[AllReduceRunner]] = {}

        self._inner_pipe, self._outer_pipe = mp.Pipe(duplex=True)  # a control pipe used to communicate with daemon

        self._allow_state_sharing = mp.Value(ctypes.c_bool, 0)
        self._state_sharing_priority = mp.Value(ctypes.c_double, 0)
        self._global_epoch = mp.Value(ctypes.c_double, 0)

        if allow_state_sharing is None:
            allow_state_sharing = not client_mode and not auxiliary
        self.allow_state_sharing = allow_state_sharing
        self.declare_state_period = declare_state_period
        self.state_compression = state_compression
        self.tensor_infos = tensor_infos

        self._background_tasks = set()

        self._ready = MPFuture()
        # note: we create a background thread weakref and with daemon=True to ensure garbage collection
        background_fetcher = threading.Thread(
            daemon=True,
            target=_background_thread_fetch_current_state,
            args=[self.serializer, self._outer_pipe, weakref.WeakMethod(self.get_current_state)],
        )
        background_fetcher.start()
        if start:
            self.run_in_background(await_ready=True)

    @property
    def allow_state_sharing(self) -> bool:
        """If set to True, other peers can download this peer's state."""
        return bool(self._allow_state_sharing.value)

    @allow_state_sharing.setter
    def allow_state_sharing(self, value: bool):
        if value and self.client_mode:
            raise ValueError("Cannot allow state sharing: averager in client mode cannot share its state")
        else:
            old_value, self._allow_state_sharing.value = self._allow_state_sharing.value, value
            if value != old_value:
                self._outer_pipe.send(("_trigger_declare_load_state", [], {}))

    @property
    def state_sharing_priority(self) -> float:
        """Others will preferentially downloading state from peers with highest priority."""
        return float(self._state_sharing_priority.value)

    @state_sharing_priority.setter
    def state_sharing_priority(self, value: float):
        if value and self.client_mode:
            raise ValueError("State sharing priority is unused: averager in client mode cannot share its state")
        else:
            old_value, self._state_sharing_priority.value = self._state_sharing_priority.value, value
            if self.allow_state_sharing and value != old_value:
                self._outer_pipe.send(("_trigger_declare_load_state", [], {}))

    @property
    def global_epoch(self) -> float:
        """Global epoch used in AR operations that require to sync seeds."""
        return float(self._global_epoch.value)

    @global_epoch.setter
    def global_epoch(self, value: float):
        self._global_epoch.value = value

    async def _trigger_declare_load_state(self):
        # note: previously tried to set mp.Event instead of this. Awaiting it in executor caused degradation in py39
        self._state_updated.set()

    @property
    def peer_id(self) -> PeerID:
        return self.dht.peer_id

    @property
    def request_timeout(self):
        return self._matchmaking.request_timeout

    def _handle_done_callback(self, task: asyncio.Task):
        self._background_tasks.discard(task)
        try:
            task.result()
        except:  # noqa: E722 catching all exceptions here
            pass

    def run(self):
        """Run averager function in a background thread; this is needed to avoid a heisenbug with broken OMP on fork.
        Turns out, using a non-main thread creates a separate OMP pool that works even if the original pool is corrupted.
        Read more: https://github.com/pytorch/pytorch/issues/17199
        """
        thread = threading.Thread(target=self._run_internal, daemon=True)
        thread.start()
        thread.join()

    def _run_internal(self):
        """Serve DecentralizedAverager forever. This function will not return until the averager is shut down."""
        loop = switch_to_uvloop()
        # initialize asyncio synchronization primitives in this event loop

        pipe_semaphore = asyncio.Semaphore(value=0)
        loop.add_reader(self._inner_pipe.fileno(), pipe_semaphore.release)

        async def _run():
            try:
                self._p2p = await self.dht.replicate_p2p()
                if not self.client_mode:
                    await self.add_p2p_handlers(self._p2p, namespace=self.prefix)
                else:
                    logger.debug("The averager is running in client mode")

                matchmaking_cls = build_cls(self.matchmaking_class, self.matchmaking_kwargs, partial_init=True)
                self._matchmaking = matchmaking_cls(
                    dht=self.dht,
                    prefix=self.prefix,
                )

                if not self.client_mode:
                    asyncio.create_task(self._declare_for_download_periodically())

                self._state_updated = asyncio.Event()
                self._pending_groups_registered = asyncio.Event()
            except Exception as e:
                # Loglevel is DEBUG since normally the exception is propagated to the caller
                logger.debug(e, exc_info=True)
                self._ready.set_exception(e)
                return
            self._ready.set_result(None)

            while True:
                try:
                    await asyncio.wait_for(pipe_semaphore.acquire(), timeout=self.request_timeout)
                except asyncio.TimeoutError:
                    pass
                if not self._inner_pipe.poll():
                    continue
                try:
                    method, args, kwargs = self._inner_pipe.recv()
                except (OSError, ConnectionError, RuntimeError) as e:
                    logger.error(e, exc_info=True)
                    await asyncio.sleep(self.request_timeout)
                    continue
                task = asyncio.create_task(getattr(self, method)(*args, **kwargs))

                if method == "_shutdown":
                    await task
                    break

                self._background_tasks.add(task)
                task.add_done_callback(self._handle_done_callback)

        loop.run_until_complete(_run())

    def run_in_background(self, await_ready: bool = True, timeout: float | None = None) -> None:
        """Starts averager in a background process. if await_ready, this method will wait until background dht
        is ready to process incoming requests or for :timeout: seconds max.
        """
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        self.start()
        if await_ready:
            self.wait_until_ready(timeout)

    def wait_until_ready(self, timeout: float | None = None) -> None:
        self._ready.result(timeout=timeout)

    def shutdown(self) -> None:
        """Shut down the averager process."""
        if self.is_alive():
            self._outer_pipe.send(("_shutdown", [self.shutdown_timeout], {}))  # shut down the daemon process
            self._inner_pipe.send(("_SHUTDOWN", None))  # shut down background thread in master
            self.join(self.shutdown_timeout)
            if self.is_alive():
                logger.warning("Averager did not shut down within the grace period; terminating it the hard way")
                self.terminate()
        else:
            logger.error(
                "Averager shutdown has no effect: the process is already not alive",
                exc_info=True,
            )

    async def _shutdown(self, timeout: float | None) -> None:
        if not self.client_mode:
            await self.remove_p2p_handlers(self._p2p, namespace=self.prefix)

        if self._p2p is not None:
            await self._p2p.shutdown()

        for future in self._running_groups.values():
            if future.done() and not future.cancelled():
                try:
                    runner = future.result()
                    runner.finalize(cancel=True)
                except Exception:
                    pass
            elif not future.done():
                future.cancel()

        for task in self._background_tasks:
            task.cancel()

    def __del__(self):
        if self._parent_pid == os.getpid() and self.is_alive():
            self.shutdown()

    def step(
        self,
        gather: GatheredData | None = None,
        scheduled_time: DHTExpiration | None = None,
        weight: float | None = None,
        timeout: float | None = None,
        allow_retries: bool = False,
        require_trigger: bool = False,
        wait: bool = True,
    ) -> dict[PeerID, GatheredData] | None | StepControl:
        """Set up the averager to look for a group and run one round of averaging.

        Args:
            gather (GatheredData | None, optional): Optionally send this information
                to all peers in the next group and gather it from every groupmate
                (this operation is known as all-gather). The gathered data will be
                available as the output of this function. Defaults to None.
            scheduled_time (DHTExpiration | None, optional): When matchmaking, assume
                that all-reduce will begin at this moment. By default, schedule
                all-reduce current time plus min_matchmaking_time seconds. Defaults to None.
            weight (float | None, optional): Averaging weight for this peer, int or
                float, must be strictly positive. Defaults to None.
            timeout (float | None, optional): If averager was unable to *find* a group
                in this many seconds, consider allreduce failed. Defaults to None.
            allow_retries (bool, optional): If averager fails to run one round of
                allreduce, this option will allow it to try again within the specified
                timeout. Defaults to False.
            require_trigger (bool, optional): If True, await for user to call
                .allow_allreduce() before running all-reduce. Defaults to False.
            wait (bool, optional): If True, return when finished. Otherwise return
                StepControl and run in background. Defaults to True.

        Returns:
            dict[PeerID, GatheredData] | None | StepControl: On success, update averaged_tensors and return group info; on failure, return None.
        """
        if self.mode == AveragingMode.AUX and weight is not None:
            logger.warning("Averager is running in auxiliary mode, weight is unused")
        if scheduled_time is None:
            scheduled_time = get_dht_time() + self.matchmaking_kwargs["min_matchmaking_time"]
        if weight is None:
            weight = float(self.mode != AveragingMode.AUX)
        deadline = get_dht_time() + timeout if timeout is not None else float("inf")
        assert isinstance(weight, (int, float)) and weight >= 0, f"Expected a positive int/float, got {type(weight)}"
        assert not (wait and require_trigger), "Non-asynchronous step cannot wait for trigger (use wait=False)"
        assert scheduled_time < deadline, "Scheduled start time does not fit within timeout"

        user_data_for_gather = self.serializer.dumps(gather)  # serialize here to avoid imports in the averager process
        data_for_gather = self.serializer.dumps([self.bandwidth, self.mode.value, user_data_for_gather])
        step = StepControl(
            scheduled_time=scheduled_time,
            deadline=deadline,
            allow_retries=allow_retries,
            weight=weight,
            data_for_gather=data_for_gather,
        )

        future_for_init = MPFuture()
        self._outer_pipe.send(("_step", [], dict(step=step, future_for_init=future_for_init)))
        step.attach(*future_for_init.result())

        if not require_trigger:
            step.allow_allreduce()
        return step.result() if wait else step

    async def _step(self, *, step: StepControl, future_for_init: MPFuture):
        try:
            trigger, cancel = MPFuture(), MPFuture()
            step.attach(trigger, cancel)
            future_for_init.set_result((trigger, cancel))

            async def find_peers_or_notify_cancel():
                group_info = await self._matchmaking.look_for_group(step, global_step=int(self.global_epoch))
                if not step.triggered:
                    step.stage = AveragingStage.AWAITING_TRIGGER
                    await step.wait_for_trigger()
                return group_info

            while not step.done():
                try:
                    self._pending_groups_registered.clear()
                    step.stage = AveragingStage.LOOKING_FOR_GROUP
                    matchmaking_task = asyncio.create_task(find_peers_or_notify_cancel())
                    check_cancel_task = asyncio.create_task(step.wait_for_cancel())
                    self._background_tasks.add(matchmaking_task)
                    self._background_tasks.add(check_cancel_task)
                    matchmaking_task.add_done_callback(self._handle_done_callback)
                    check_cancel_task.add_done_callback(self._handle_done_callback)

                    await asyncio.wait({matchmaking_task, check_cancel_task}, return_when=asyncio.FIRST_COMPLETED)
                    if step.cancelled():
                        matchmaking_task.cancel()
                        raise asyncio.CancelledError()
                    else:
                        check_cancel_task.cancel()

                    group_info = await matchmaking_task

                    if group_info is None:
                        raise AllreduceException("Averaging step failed: could not find a group")

                    with self._register_allreduce_group(group_info):
                        step.stage = AveragingStage.RUNNING_ALLREDUCE

                        step.set_result(
                            await asyncio.wait_for(
                                self._aggregate_with_group(
                                    group_info,
                                    tensor_infos=self.tensor_infos,
                                    weight=step.weight,
                                    **self.allreduce_kwargs,
                                ),
                                timeout=self._allreduce_timeout,
                            )
                        )
                        # averaging is finished, loop will now exit
                except (
                    AllreduceException,
                    MatchmakingException,
                    AssertionError,
                    StopAsyncIteration,
                    asyncio.CancelledError,
                    asyncio.InvalidStateError,
                    P2PHandlerError,
                    P2PDaemonError,
                ) as e:
                    if step.done() or not step.allow_retries or get_dht_time() >= step.deadline:
                        if not step.cancelled():
                            logger.error(f"{e} during {step.stage.name}", exc_info=True)
                        if not step.done():
                            # keeping the traceback/cause/context pins frames and leaks memory
                            e.__traceback__ = e.__cause__ = e.__context__ = None
                            step.set_exception(e)
                    else:
                        logger.warning(
                            f"{self.__class__.__name__} caught {repr(e)} during {step.stage.name}, retrying"
                        )

        except BaseException as e:
            if not step.done():
                # keeping the traceback/cause/context pins frames and leaks memory
                e.__traceback__ = e.__cause__ = e.__context__ = None
                step.set_exception(e)
            raise
        finally:
            step.stage = AveragingStage.FINISHED
            if not step.done():
                step.set_exception(
                    RuntimeError(
                        "Internal sanity check failed: averager.step left future pending."
                        " Please report this to hivemind issues."
                    )
                )
            if self.mode == AveragingMode.AUX:
                # Auxiliary peers reduce every part of every round; without a trim the
                # freed round buffers stay cached in allocator arenas and RSS ratchets.
                gc.collect()
                malloc_trim = _libc_malloc_trim()
                if malloc_trim is not None:
                    malloc_trim(0)
                logger.debug("Auxiliary averager trimmed heap after averaging round")

    @contextlib.contextmanager
    def _register_allreduce_group(self, group_info: GroupInfo):
        """Register a given group for one or more all-reduce rounds."""
        try:
            self._running_groups[group_info.group_id] = asyncio.Future()
            self._pending_groups_registered.set()
            yield
        finally:
            maybe_future = self._running_groups.pop(group_info.group_id, None)
            if maybe_future is not None and not maybe_future.done():
                logger.warning(f"All-reduce group {group_info.group_id} did not finish.")
            self._pending_groups_registered.clear()

    async def _aggregate_with_group(self, group_info: GroupInfo, min_vector_size: int, **kwargs) -> GatheredData:
        """Run aggregation in a given group and update tensors in place, return gathered metadata."""
        try:
            bandwidths, mode_ids, user_gathered_bytes = zip(*map(self.serializer.loads, group_info.gathered))
            user_gathered = dict(zip(group_info.peer_ids, map(self.serializer.loads, user_gathered_bytes)))
            modes = tuple(map(AveragingMode, mode_ids))

            # compute optimal part sizes from peer bandwidths; TODO: replace with proper load balancing
            download_bandwidths = [
                thr if mode != AveragingMode.CLIENT else 0.0 for thr, mode in zip(bandwidths, modes)
            ]
            peer_fractions = await asyncio.get_event_loop().run_in_executor(
                None, load_balance_peers, self.total_size, download_bandwidths, min_vector_size
            )

            async with enter_asynchronously(self.get_tensors()) as local_tensors:
                await self._run_allreduce_inplace_(
                    local_tensors, group_info, peer_fractions=peer_fractions, modes=modes, **kwargs
                )
                return user_gathered
        except BaseException as e:
            if isinstance(e, Exception):
                logger.error(e, exc_info=True)
            raise MatchmakingException(f"Unable to run All-Reduce: {type(e).__name__}: {e}")

    async def _run_allreduce_inplace_(
        self,
        tensors: Sequence[torch.Tensor],
        group_info: GroupInfo,
        group_id: bytes | None = None,
        **kwargs,
    ):
        """Run one allreduce process to average tensors inplace. Can be called more than a few times in one aggregation process."""
        group_id = group_info.group_id if group_id is None else group_id

        runner_cls = build_cls(self._ar_runner_class, partial_init=True)

        runner = runner_cls(
            p2p=self._p2p,
            servicer_type=type(self),
            prefix=self.prefix,
            tensors=tensors,
            group_id=group_id,
            ordered_peer_ids=group_info.peer_ids,
            **kwargs,
        )
        assert group_id in self._running_groups, f"Group id {group_id} was not registered in _register_allreduce_group"
        self._running_groups[group_id].set_result(runner)

        if runner.modes[group_info.peer_ids.index(self.peer_id)] != AveragingMode.AUX:
            async for tensor, update in azip(as_aiter(*tensors), runner):
                tensor.add_(update, alpha=self._averaging_alpha)
            self.last_updated = get_dht_time()
        else:
            async for _ in runner:
                raise ValueError("aux peers should not receive averaged tensors")

    @contextlib.contextmanager
    def get_tensors(self) -> Sequence[torch.Tensor]:
        """A contextmanager that gives user access to averaged tensors.

        It is guaranteed that the averager will not modify tensors while this context is active.
        Please do not modify the yielded tensors in-place after the context is released.
        """
        with self.lock_averaged_tensors:
            yield self._averaged_tensors

    async def rpc_join_group(
        self,
        request: averaging_pb2.JoinRequest,
        context: P2PContext,
    ) -> AsyncIterator[averaging_pb2.MessageFromLeader]:
        """Accept or reject a join request from another averager; if accepted, run him through allreduce steps."""
        async for response in self._matchmaking.rpc_join_group(request, context):
            yield response

    async def rpc_aggregate_part(
        self,
        stream: AsyncIterator[averaging_pb2.AveragingData],
        context: P2PContext,
    ) -> AsyncIterator[averaging_pb2.AveragingData]:
        """A groupmate sends us a part of his tensor; we should average it with other peers and return the result."""
        request = await anext(stream)
        if request.group_id not in self._running_groups:
            # this handles a special case when leader accepted us to group AND began allreduce right away,
            # but his response with group_id was delayed and other peers got to us first
            await self._pending_groups_registered.wait()

        future = self._running_groups.get(request.group_id)
        if future is None:
            yield averaging_pb2.AveragingData(code=averaging_pb2.BAD_GROUP_ID)
            return

        group = await future
        async for message in group.rpc_aggregate_part(achain(as_aiter(request), stream), context):
            yield message

    async def _declare_for_download_periodically(self):
        download_key = f"{self._matchmaking.group_key_manager.prefix}.all_averagers"
        sharing_was_allowed = self.allow_state_sharing
        while True:
            expiration_time = get_dht_time() + self.declare_state_period
            if self.allow_state_sharing or sharing_was_allowed:
                # notify either if sharing is allowed or if it was just switched off (to overwrite previous message)
                asyncio.create_task(
                    asyncio.wait_for(
                        self.dht.store(
                            download_key,
                            subkey=self.peer_id.to_bytes(),
                            value=self.state_sharing_priority if self.allow_state_sharing else None,
                            expiration_time=expiration_time,
                            return_future=True,
                        ),
                        timeout=expiration_time - get_dht_time(),
                    )
                )
                sharing_was_allowed = self.allow_state_sharing

            # report again either in state_declare_period or after the field was changed by the user
            self._state_updated.clear()
            try:
                await asyncio.wait_for(self._state_updated.wait(), timeout=max(0.0, expiration_time - get_dht_time()))
            except asyncio.TimeoutError:
                pass

    async def rpc_download_state(
        self,
        _request: averaging_pb2.DownloadRequest,
        _context: P2PContext,
        chunk_size_bytes: int = 2**19,
    ) -> AsyncIterator[averaging_pb2.DownloadData]:
        """Get the up-to-date trainer state from a peer.

        The state consists of two parts: (serialized_metadata, tensors)

        - serialized_metadata is a small serialized bytestring meant to store scalars and hyperparameters
        - tensors is a sequence of pytorch tensors that represent model parameters or optimizer statistics
        """
        if not self.allow_state_sharing:
            return

        # Decline if already serving another peer
        if self._state_download_lock.locked():
            return

        def cleanup_tensors(tensors):
            """Clean up tensors including shared memory."""
            if tensors is None:
                return
            for tensor in tensors:
                if hasattr(tensor, "is_shared") and tensor.is_shared():
                    try:
                        tensor.detach_()
                        if hasattr(tensor.untyped_storage(), "_free_weak_ref"):
                            tensor.untyped_storage()._free_weak_ref()
                    except:  # noqa: E722 catching all exceptions here
                        pass

        metadata = tensors = infos = None
        try:
            async with self._state_download_lock:
                try:
                    metadata, tensors, infos = await self._get_current_state_from_host_process()
                    if infos is None:
                        infos = [CompressionInfo.from_tensor(tensor, key=i) for i, tensor in enumerate(tensors)]
                    assert len(tensors) == len(infos)

                    for tensor, info in zip(tensors, infos):
                        for part in split_for_streaming(
                            self.state_compression.compress(tensor, info, allow_inplace=False),
                            chunk_size_bytes=chunk_size_bytes,
                        ):
                            if metadata is not None:
                                yield averaging_pb2.DownloadData(tensor_part=part, metadata=metadata)
                                metadata = None
                            else:
                                yield averaging_pb2.DownloadData(tensor_part=part)
                finally:
                    cleanup_tensors(tensors)
        finally:
            cleanup_tensors(tensors)

    def get_current_state(self) -> tuple[Any, Sequence[torch.Tensor], Sequence[CompressionInfo]]:
        """Get current state and send it to a peer. Executed in the host process. Meant to be overridden.

        Returns:
            tuple[Any, Sequence[torch.Tensor], Sequence[CompressionInfo]]: (small metadata, sequence of torch tensors)

        Note:
            metadata must be seriablizable with self.serializer (default = MSGPackSerializer)
        """
        with self.get_tensors() as tensors:
            return dict(group_key=self.get_group_bits()), tensors, self.tensor_infos

    async def _get_current_state_from_host_process(self):
        """Executed in the averager process inside rpc_download_state"""
        future = MPFuture()
        self._inner_pipe.send(("_TRIGGER_GET_CURRENT_STATE", future))
        return await future

    def load_state_from_peers(
        self,
        target_peer: PeerID | None = None,
        wait: bool = True,
        timeout: float | None = None,
        load_state_sort_strategy: PeerSortStrategy = "uid",
    ) -> tuple[Any, Sequence[torch.Tensor], PeerID] | None:
        """Try to download the latest optimizer state one of the existing peer.

        Returns:
            On success, return a 2-tuple with (metadata, tensors), where
            - metadata is a small object containing metadata (e.g. hyperparameters, scalars, etc)
            - tensors is a sequence of pytorch tensors meant to contain peer's model weights and optimizer statistics

        The exact contents of both metadata and tensors are determined by get_current_state method
        """
        future = MPFuture()
        self._outer_pipe.send(
            (
                "_load_state_from_peers",
                [],
                dict(
                    target_peer=target_peer,
                    timeout=timeout,
                    future=future,
                    load_state_sort_strategy=load_state_sort_strategy,
                ),
            )
        )
        return future.result(timeout=timeout) if wait else future

    async def _load_state_from_peers(
        self,
        target_peer: PeerID | None,
        future: MPFuture,
        timeout: float | None = None,
        load_state_sort_strategy: PeerSortStrategy = "uid",
    ):
        if timeout is not None:
            timeout = self.next_chunk_timeout if self.next_chunk_timeout is not None else self.request_timeout

        try:
            key_manager = self._matchmaking.group_key_manager
            peer_priority, _ = self.dht.get(f"{key_manager.prefix}.all_averagers", latest=True) or ({}, None)
            peer_priority = {
                PeerID(peer_id): (float(info.value), random.random())  # using randomness as a tie breaker
                for peer_id, info in peer_priority.items()
                if isinstance(info, ValueWithExpiration) and isinstance(info.value, (float, int))
            }

            if not isinstance(peer_priority, dict) or len(peer_priority) == 0:
                logger.info(f"Averager could not load state from peers: peer dict empty or corrupted {peer_priority}")
                future.set_result(None)
                return

            if load_state_sort_strategy == "priority":
                sorted_peers = sorted(peer_priority.keys(), key=peer_priority.get, reverse=True)
            elif load_state_sort_strategy == "uid":
                sorted_peers = self._sort_peers_by_uid(key_manager, peer_priority)
                if sorted_peers is None:
                    future.set_result(None)
                    return

            if target_peer:
                sorted_peers.insert(0, target_peer)

            metadata = None
            for peer in sorted_peers:
                if peer != self.peer_id:
                    t0 = time.monotonic()
                    logger.info(f"Downloading parameters from peer {peer}")
                    try:
                        stub = self.get_stub(self._p2p, peer, namespace=self.prefix)
                        stream = await stub.rpc_download_state(averaging_pb2.DownloadRequest())
                        current_tensor_parts, tensors = [], []

                        # TODO merge this with hivemind.compression.deserialize_tensor_stream
                        async for message in aiter_with_timeout(stream, timeout=timeout):
                            if message.metadata:
                                metadata = self.serializer.loads(message.metadata)
                            if message.tensor_part.dtype and current_tensor_parts:
                                # tensor_part.dtype indicates the start of the new tensor, so we should wrap up this one
                                tensors.append(deserialize_torch_tensor(combine_from_streaming(current_tensor_parts)))
                                current_tensor_parts = []
                            current_tensor_parts.append(message.tensor_part)

                        if current_tensor_parts:
                            tensors.append(deserialize_torch_tensor(combine_from_streaming(current_tensor_parts)))

                        if not metadata:
                            logger.debug(f"Peer {peer} did not send its state")
                            continue

                        t1 = time.monotonic()
                        logger.info(f"Finished downloading state in {t1 - t0:.3f}s from {peer}")

                        # Check if any tensor contains NaN or inf values
                        has_nans = any(not torch.isfinite(t).all() for t in tensors)
                        if has_nans:
                            logger.error("Failed to load state from peer. Downloaded state contains invalid values")

                        future.set_result((metadata, tensors, peer))
                        return
                    except Exception as e:
                        logger.error(f"Failed to load state from {peer} - {repr(e)}", exc_info=True)

        finally:
            if not future.done():
                future.set_result(None)

    def _sort_peers_by_uid(
        self,
        key_manager,
        peer_priority: dict[PeerID, tuple[float, float]],
    ) -> list[PeerID] | None:
        stage_name = key_manager.prefix.split("_")[0]
        expert_key = f"{stage_name}.0."
        response = self.dht.get(expert_key, latest=True)

        peer_uid_map = {}
        if isinstance(response, ValueWithExpiration) and isinstance(response.value, dict):
            items = dict(response.value.items())
            for item in items.values():
                peer_id = item.value[1]
                peer_uid = item.value[0]
                peer_uid_idx = int(peer_uid.split(".")[-1])
                peer_uid_map[peer_id] = peer_uid_idx
        else:
            logger.info(f"Averager could not load state from peers: peer dict empty or corrupted {response}")
            return None

        return sorted(peer_priority.keys(), key=lambda p: peer_uid_map.get(str(p), float("inf")))

    def get_group_bits(self, wait: bool = True):
        """Get averager's current group key bits (without prefix).

        Args:
            wait (bool, optional): If True, return bits immediately. Otherwise return awaitable MPFuture. Defaults to True.
        """
        future = MPFuture()
        self._outer_pipe.send(("_get_group_bits", [], dict(future=future)))
        return future.result() if wait else future

    async def _get_group_bits(self, future: MPFuture):
        future.set_result(self._matchmaking.group_key_manager.group_bits)

    def set_group_bits(self, group_bits: str, wait: bool = True):
        """Set group bits to be used in averager's group key.

        Args:
            group_bits (str): Group bits (string of '0' or '1') to be used in averager's group key.
            wait (bool, optional): If True, wait until the update is confirmed by the averager. Otherwise return immediately. Defaults to True.
        """
        future = MPFuture()
        assert all(bit in "01" for bit in group_bits)
        self._outer_pipe.send(("_set_group_bits", [], dict(group_bits=group_bits, future=future)))
        return future.result() if wait else future

    async def _set_group_bits(self, group_bits: str, future: MPFuture):
        try:
            self._matchmaking.group_key_manager.group_bits = group_bits
            return future.set_result(None)
        except Exception as e:
            if not future.done():
                future.set_exception(e)


_state_fetch_lock = threading.Lock()


def _background_thread_fetch_current_state(
    serializer: SerializerBase,
    pipe: mp.connection.Connection,
    get_current_state_ref: weakref.WeakMethod,
):
    """Fetch the averager state when asked by peers.

    Executed in the host process as a background thread.

    Args:
        serializer (SerializerBase): A serializer with which to convert metadata into bytes.
        pipe (mp.connection.Connection): DecentralizedAverager's control pipe (from host process side).
        get_current_state_ref (weakref.WeakMethod): A WeakMethod wrapped around DecentralizedAverager.get_current_state (instance-bound).
    """
    while True:
        try:
            trigger, future = pipe.recv()
        except BaseException as e:
            logger.debug(f"Averager background thread finished: {repr(e)}")
            break

        if trigger == "_SHUTDOWN":
            break

        assert trigger == "_TRIGGER_GET_CURRENT_STATE"
        with _state_fetch_lock:
            try:
                get_current_state = get_current_state_ref()
                if get_current_state is None:
                    break
                state = get_current_state()
                assert 0 < len(state) <= 3
                if len(state) != 3:
                    state = tuple(state + (None,) * (3 - len(state)))
                state_metadata, state_tensors, tensor_infos = state
                del get_current_state

                state_metadata = serializer.dumps(state_metadata)
                state_tensors = tuple(
                    tensor.cpu().detach().requires_grad_(tensor.requires_grad) for tensor in state_tensors
                )
                # note: we cast tensors to CPU on host side to avoid initializing cuda in the guest process
                future.set_result((state_metadata, state_tensors, tensor_infos))
            except BaseException as e:
                future.set_exception(e)
                logger.warning(e)
                continue


def compute_schema_hash(tensors: Sequence[torch.Tensor]) -> bytes:
    """A hash that describes follower's tensor shapes, dtypes, devices, but not the actual values."""
    schema_dicts = [
        {
            field_name: str(field_value)
            for field_name, field_value in asdict(TensorDescriptor.from_tensor(tensor)).items()
        }
        for tensor in tensors
    ]
    return DHTID.generate(source=schema_dicts).to_bytes()
