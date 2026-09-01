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

import asyncio
import ctypes
import multiprocessing as mp
import time

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Optional

import torch

from agora_server.hivemind.compression import (
    deserialize_tensor_stream,
    deserialize_torch_tensor,
    serialize_torch_tensor,
)
from agora_server.hivemind.dht import DHT
from agora_server.hivemind.moe.server.module_backend import ModuleBackend
from agora_server.hivemind.moe.server.task_pool import TaskPool
from agora_server.hivemind.p2p import P2PContext, ServicerBase
from agora_server.hivemind.p2p.p2p_daemon import DEFAULT_MAX_MSG_SIZE, P2P
from agora_server.hivemind.proto import runtime_pb2
from agora_server.hivemind.utils import MPFuture, MSGPackSerializer, get_logger, nested_flatten
from agora_server.hivemind.utils.asyncio import amap_in_executor, switch_to_uvloop
from agora_server.hivemind.utils.streaming import split_for_streaming
from agora_server.hivemind.utils.tensor_descr import BatchTensorDescriptor


if TYPE_CHECKING:
    # Type-only imports: these live in agora_server.core. Importing them at runtime would invert
    # the hivemind->core layering (and risk a circular import).
    from agora_server.core.server.activation_cache import ActivationCache
    from agora_server.core.server.w2w_coordinator import NextHopLearner
    from agora_server.core.server.w2w_dataplane import DirectW2WDriver


logger = get_logger(__name__)

_HEARTBEAT_INTERVAL_S = 10.0


class ConnectionHandler(mp.context.ForkProcess, ServicerBase):
    """
    A process that accepts incoming requests to experts and submits them into the corresponding TaskPool.

    :note: ConnectionHandler is designed so as to allow using multiple handler processes for the same port
    :param dht: a running hivemind.dht.DHT, used to let other peers connect to this one
    :param module_backends: a dict [UID -> ModuleBackend] with all active experts
    :param activation_cache: optional cross-process activation cache. When provided, the forward RPC
        stashes each microbatch's input activation and the backward RPC serves the cached input back
        so the trainer sends only gradients. A microbatch's forward and backward are load-balanced
        onto *different* handler processes of the same worker, so the cache cannot be process-local:
        it is created once in the parent ``Server`` and shared across all handlers via an ``mp.Manager``.
        When ``None`` (flag off), every RPC behaves exactly as before.
    :param handler_index: this handler's slot in ``heartbeat``, in ``Server.conn_handlers`` order
    :param heartbeat: optional lock-free shared array of per-handler ``time.monotonic()`` stamps.
        Written every ``_HEARTBEAT_INTERVAL_S`` from a task on the handler's own event loop and read
        by the parent ``Server``'s liveness reporter, so a wedged loop surfaces as a growing age.
        Each handler must write only its own slot -- single-writer slots are what make the array
        safe without a lock.
    :param w2w_coordinator: optional cross-process hop learner (next-hop coordination plane). When
        provided, the forward AND backward RPCs parse the microbatch metadata and, if a w2w trainer
        taught a next and/or prev hop, log + count it. Purely observational: it never retains or acts
        on the hop and does not change the response, data path, or timing. Shared across handlers like
        ``activation_cache``. When ``None`` (flag off), every RPC behaves exactly as before.
    """

    def __init__(
        self,
        dht: DHT,
        module_backends: dict[str, ModuleBackend],
        *,
        balanced: bool = True,
        shutdown_timeout: float = 3,
        handler_index: int = 0,
        heartbeat: ctypes.Array | None = None,
        start: bool = False,
        activation_cache: Optional["ActivationCache"] = None,
        w2w_coordinator: Optional["NextHopLearner"] = None,
        w2w_forward_driver: Optional["DirectW2WDriver"] = None,
    ):
        super().__init__()
        self.dht, self.module_backends = dht, module_backends
        self.balanced, self.shutdown_timeout = balanced, shutdown_timeout
        self.handler_index, self.heartbeat = handler_index, heartbeat
        self.activation_cache = activation_cache
        self.w2w_coordinator = w2w_coordinator
        self.w2w_forward_driver = w2w_forward_driver
        self._p2p: P2P | None = None
        # Strong refs to in-flight w2w handler tasks: asyncio keeps only a weak ref to a create_task'd
        # task, so without this an in-flight handle_forward/handle_backward can be GC'd mid-run
        # ("Task was destroyed but it is pending!") and the microbatch is silently dropped.
        self._w2w_bg_tasks: set = set()

        self._inner_pipe, self._outer_pipe = mp.Pipe(duplex=False)
        self.ready = MPFuture()

        if start:
            self.run_in_background(await_ready=True)

    def _spawn_w2w(self, coro):
        """create_task the w2w handler coroutine and retain a strong ref until it finishes."""
        task = asyncio.create_task(coro)
        self._w2w_bg_tasks.add(task)
        task.add_done_callback(self._w2w_bg_tasks.discard)
        return task

    async def _stamp_heartbeat(self, heartbeat: ctypes.Array) -> None:
        """Stamp this handler's heartbeat slot forever. Runs as a task on the handler's own event
        loop: a wedged loop stops the stamps, which is exactly the failure the parent's liveness
        reporter exists to expose (a thread in this process would keep stamping through the wedge)."""
        while True:
            heartbeat[self.handler_index] = time.monotonic()
            await asyncio.sleep(_HEARTBEAT_INTERVAL_S)

    def run(self):
        torch.set_num_threads(1)
        loop = switch_to_uvloop()
        stop = asyncio.Event()
        loop.add_reader(self._inner_pipe.fileno(), stop.set)

        async def _run():
            receipt_servicer = None
            try:
                self._p2p = await self.dht.replicate_p2p()
                await self.add_p2p_handlers(self._p2p, balanced=self.balanced)
                driver = self.w2w_forward_driver
                if (
                    driver is not None
                    and getattr(driver, "trainerless", False)
                    and getattr(driver, "receipt_mailbox", None) is not None
                ):
                    # Trainerless: receipts for this worker's reservations. Registered
                    # balanced; the servicer only stamps the shared mailbox, so any
                    # process may receive them.
                    from agora_server.core.server.w2w_dataplane import W2W_COORD_NAMESPACE, W2W_RECEIPT_NAMESPACE
                    from agora_server.core.server.w2w_local_router import make_receipt_servicer

                    receipt_servicer = make_receipt_servicer(driver.receipt_mailbox)
                    await receipt_servicer.add_p2p_handlers(self._p2p, namespace=W2W_RECEIPT_NAMESPACE, balanced=True)
                    # Origin heads serve batch reports here too: the ledger is
                    # manager-backed so any process can serve it, and registering from
                    # the main process after the handlers are up deadlocks the daemon.
                    if driver.origin_ledger is not None:
                        ledger_servicer = driver.origin_ledger.servicer
                        await ledger_servicer.add_p2p_handlers(self._p2p, namespace=W2W_COORD_NAMESPACE, balanced=True)
                self.ready.set_result(None)
            except Exception as e:
                logger.error("ConnectionHandler failed to start:", exc_info=True)
                self.ready.set_exception(e)
                return

            heartbeat_task = (
                None if self.heartbeat is None else asyncio.create_task(self._stamp_heartbeat(self.heartbeat))
            )
            try:
                await stop.wait()
            finally:
                if heartbeat_task is not None:
                    heartbeat_task.cancel()
                await self.remove_p2p_handlers(self._p2p)
                if receipt_servicer is not None:
                    from agora_server.core.server.w2w_dataplane import W2W_COORD_NAMESPACE, W2W_RECEIPT_NAMESPACE

                    await receipt_servicer.remove_p2p_handlers(self._p2p, namespace=W2W_RECEIPT_NAMESPACE)
                    driver = self.w2w_forward_driver
                    if driver is not None and driver.origin_ledger is not None:
                        await driver.origin_ledger.servicer.remove_p2p_handlers(
                            self._p2p, namespace=W2W_COORD_NAMESPACE
                        )
                if self._p2p is not None:
                    await self._p2p.shutdown()

        try:
            loop.run_until_complete(_run())
        except KeyboardInterrupt:
            logger.debug("Caught KeyboardInterrupt, shutting down")

    def run_in_background(self, await_ready: bool = True, timeout: float | None = None) -> None:
        """
        Starts ConnectionHandler in a background process. If :await_ready:, this method will wait until
        it is ready to process incoming requests or for :timeout: seconds max.
        """
        self.start()
        if await_ready:
            self.wait_until_ready(timeout)

    def wait_until_ready(self, timeout: float | None = None) -> None:
        self.ready.result(timeout=timeout)

    def shutdown(self):
        if self.is_alive():
            self._outer_pipe.send("_shutdown")
            self.join(self.shutdown_timeout)
            if self.is_alive():
                logger.warning(
                    "ConnectionHandler did not shut down within the grace period; terminating it the hard way"
                )
                self.terminate()
        else:
            logger.warning("ConnectionHandler shutdown had no effect, the process is already dead")

    async def rpc_info(self, request: runtime_pb2.ExpertUID, context: P2PContext) -> runtime_pb2.ExpertInfo:
        module_info = self.module_backends[request.uid].get_info()
        return runtime_pb2.ExpertInfo(serialized_info=MSGPackSerializer.dumps(module_info))

    async def _gather_inputs(
        self, requests: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext
    ) -> tuple[str, bytes, list[torch.Tensor]]:
        expert_uid = None
        metadata = b""

        async def _mark_first(
            reqs: AsyncIterator[runtime_pb2.ExpertRequest],
        ) -> AsyncIterator[runtime_pb2.ExpertRequest]:
            # Capture uid + first-message metadata in stream order on the event loop. Doing this as a
            # side effect inside the threadpool-executor _unpack would race: _unpack runs out of order,
            # so a later (empty-metadata) request could overwrite the real first-message metadata.
            nonlocal expert_uid, metadata
            async for req in reqs:
                if expert_uid is None:
                    expert_uid, metadata = req.uid, req.metadata
                elif expert_uid != req.uid:
                    raise ValueError("Expert uids differ in one request")
                yield req

        tensors_stream = amap_in_executor(lambda req: req.tensors, _mark_first(requests))
        inputs = await deserialize_tensor_stream(tensors_stream)
        return expert_uid, metadata, inputs

    async def _process_inputs(
        self,
        inputs: list[torch.Tensor],
        pool: TaskPool,
        schema: BatchTensorDescriptor | tuple[BatchTensorDescriptor, ...],
    ) -> list[runtime_pb2.Tensor]:
        return [
            serialize_torch_tensor(result, proto.compression, allow_inplace=True)
            for result, proto in zip(await pool.submit_task(*inputs), nested_flatten(schema))
        ]

    # -- activation caching -----------------------------------------

    def _cache_enabled_for(self) -> bool:
        # Caching happens iff a cache is wired into this handler. The fused-tail worker returns its
        # gradient inside the forward RPC and never receives a backward RPC, so it is simply created with
        # no cache (gated in Server.create); for it self.activation_cache is None and this returns False.
        return self.activation_cache is not None

    def _maybe_cache_forward(self, uid: str, metadata: bytes, inputs: list) -> None:
        # Stash the forward inputs keyed by metadata. SharedMemoryStore clones, so the cached copy is
        # isolated from any later aliasing/mutation of the caller's tensors.
        if self._cache_enabled_for() and metadata:
            key = self.activation_cache.key(uid, metadata)
            if key is not None:
                self.activation_cache.put(key, inputs)

    def _maybe_learn_hop(self, uid: str, metadata: bytes) -> None:
        # Observe the next/prev hop a w2w trainer taught in-band. Purely observational: learn()
        # parses + logs + counts and never acts on the hop or raises into the RPC path. Called on
        # forward (next hop, or the fused tail's prev hop) AND backward (prev hop). No-ops when no
        # coordinator is wired (flag off) and when the metadata carries no hop (legacy / terminus).
        if self.w2w_coordinator is not None:
            self.w2w_coordinator.learn(uid, metadata)

    async def _cached_backward_inputs(self, uid: str, metadata: bytes, grads: list) -> list:
        """Cache-mode backward: the wire carries only gradients; fetch the cached forward inputs and
        rebuild the ``(inputs, grads)`` flat tuple ``ModuleCollab.backward`` expects. A miss drops the
        microbatch (the trainer turns this RPC error into a dropped batch)."""
        key = self.activation_cache.key(uid, metadata)
        cached_inputs = self.activation_cache.pop(key) if key is not None else None
        if cached_inputs is None:
            self.activation_cache.record_drop()
            logger.warning(f"[ActCache] backward DROP for {uid}: cache miss (metadata absent or activation evicted)")
            raise RuntimeError(f"{uid} backward cache miss")
        return list(cached_inputs) + list(grads)

    async def rpc_forward(self, request: runtime_pb2.ExpertRequest, context: P2PContext) -> runtime_pb2.ExpertResponse:
        inputs = [deserialize_torch_tensor(tensor) for tensor in request.tensors]
        expert = self.module_backends[request.uid]
        metadata = request.metadata
        if metadata:
            logger.debug(f"{request.uid} forward carries {len(metadata)}B metadata")

        if self.w2w_forward_driver is not None:
            self._maybe_learn_hop(request.uid, metadata)
            n_forward = len(list(nested_flatten(expert.forward_schema)))
            forward_inputs = inputs[:n_forward]
            passthrough = inputs[n_forward:]
            status, reason, accepted = await self.w2w_forward_driver.accept_forward(request.uid, metadata, self._p2p)
            if accepted is not None and accepted.is_new:
                self._maybe_cache_forward(request.uid, metadata, forward_inputs)
                this_maddrs = [str(addr) for addr in await self._p2p.get_visible_maddrs()]
                self._spawn_w2w(
                    self.w2w_forward_driver.handle_forward(
                        uid=request.uid,
                        accepted=accepted,
                        forward_inputs=forward_inputs,
                        passthrough=passthrough,
                        passthrough_serialized=list(request.tensors[n_forward:]),
                        expert=expert,
                        p2p=self._p2p,
                        this_peer_id=self.dht.peer_id.to_base58(),
                        this_maddrs=this_maddrs,
                    )
                )
            return _build_w2w_push_response(status, reason)

        logger.debug(f"Processing inputs for expert {request.uid}")
        raw_outputs = tuple(await expert.forward_pool.submit_task(*inputs))
        self._maybe_cache_forward(request.uid, metadata, inputs)
        self._maybe_learn_hop(request.uid, metadata)
        serialized = [
            serialize_torch_tensor(result, proto.compression, allow_inplace=True)
            for result, proto in zip(raw_outputs, nested_flatten(expert.outputs_schema))
        ]
        return runtime_pb2.ExpertResponse(tensors=serialized, metadata=metadata)

    async def rpc_forward_stream(
        self, requests: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext
    ) -> AsyncIterator[runtime_pb2.ExpertResponse]:
        uid, metadata, inputs = await self._gather_inputs(requests, context)
        expert = self.module_backends[uid]
        if self.w2w_forward_driver is not None:
            self._maybe_learn_hop(uid, metadata)
            n_forward = len(list(nested_flatten(expert.forward_schema)))
            forward_inputs = inputs[:n_forward]
            passthrough = inputs[n_forward:]
            passthrough_serialized = [serialize_torch_tensor(t.cpu().detach()) for t in passthrough]
            status, reason, accepted = await self.w2w_forward_driver.accept_forward(uid, metadata, self._p2p)
            if accepted is not None and accepted.is_new:
                self._maybe_cache_forward(uid, metadata, forward_inputs)
                this_maddrs = [str(addr) for addr in await self._p2p.get_visible_maddrs()]
                self._spawn_w2w(
                    self.w2w_forward_driver.handle_forward(
                        uid=uid,
                        accepted=accepted,
                        forward_inputs=forward_inputs,
                        passthrough=passthrough,
                        passthrough_serialized=passthrough_serialized,
                        expert=expert,
                        p2p=self._p2p,
                        this_peer_id=self.dht.peer_id.to_base58(),
                        this_maddrs=this_maddrs,
                    )
                )
            yield _build_w2w_push_response(status, reason)
            return

        raw_outputs = tuple(await expert.forward_pool.submit_task(*inputs))
        self._maybe_cache_forward(uid, metadata, inputs)
        self._maybe_learn_hop(uid, metadata)
        output_split = [
            part
            for result, proto in zip(raw_outputs, nested_flatten(expert.outputs_schema))
            for part in split_for_streaming(
                serialize_torch_tensor(result, proto.compression, allow_inplace=True), DEFAULT_MAX_MSG_SIZE
            )
        ]

        for i, part in enumerate(output_split):
            yield runtime_pb2.ExpertResponse(tensors=[part], metadata=(metadata if i == 0 else b""))

    async def rpc_backward(
        self, request: runtime_pb2.ExpertRequest, context: P2PContext
    ) -> runtime_pb2.ExpertResponse:
        wire_tensors = [deserialize_torch_tensor(tensor) for tensor in request.tensors]
        expert = self.module_backends[request.uid]
        metadata = request.metadata
        if metadata:
            logger.debug(f"{request.uid} backward carries {len(metadata)}B metadata")
        # TODO: admit via accept_backward before popping the activation cache (also rpc_backward_stream); a BUSY/duplicate backward becomes a cache miss (follow-up PR).
        if self._cache_enabled_for() and metadata:
            inputs_and_grads = await self._cached_backward_inputs(request.uid, metadata, wire_tensors)
        else:
            inputs_and_grads = wire_tensors  # legacy: wire already carries (inputs, grads)
        self._maybe_learn_hop(request.uid, metadata)

        if self.w2w_forward_driver is not None:
            status, reason, accepted = await self.w2w_forward_driver.accept_backward(request.uid, metadata, self._p2p)
            if accepted is not None and accepted.is_new:
                self._spawn_w2w(
                    self.w2w_forward_driver.handle_backward(
                        uid=request.uid,
                        accepted=accepted,
                        inputs_and_grads=inputs_and_grads,
                        expert=expert,
                        p2p=self._p2p,
                    )
                )
            return _build_w2w_push_response(status, reason)

        return runtime_pb2.ExpertResponse(
            tensors=await self._process_inputs(inputs_and_grads, expert.backward_pool, expert.grad_inputs_schema),
            metadata=metadata,
        )

    async def rpc_backward_stream(
        self, requests: AsyncIterator[runtime_pb2.ExpertRequest], context: P2PContext
    ) -> AsyncIterator[runtime_pb2.ExpertResponse]:
        uid, metadata, wire_tensors = await self._gather_inputs(requests, context)
        expert = self.module_backends[uid]
        if self._cache_enabled_for() and metadata:
            inputs_and_grads = await self._cached_backward_inputs(uid, metadata, wire_tensors)
        else:
            inputs_and_grads = wire_tensors  # legacy: wire already carries (inputs, grads)
        self._maybe_learn_hop(uid, metadata)

        if self.w2w_forward_driver is not None:
            status, reason, accepted = await self.w2w_forward_driver.accept_backward(uid, metadata, self._p2p)
            if accepted is not None and accepted.is_new:
                self._spawn_w2w(
                    self.w2w_forward_driver.handle_backward(
                        uid=uid,
                        accepted=accepted,
                        inputs_and_grads=inputs_and_grads,
                        expert=expert,
                        p2p=self._p2p,
                    )
                )
            yield _build_w2w_push_response(status, reason)
            return

        output_split = [
            part
            for tensor in await self._process_inputs(inputs_and_grads, expert.backward_pool, expert.grad_inputs_schema)
            for part in split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE)
        ]

        for i, part in enumerate(output_split):
            yield runtime_pb2.ExpertResponse(tensors=[part], metadata=(metadata if i == 0 else b""))


def _build_w2w_push_response(status, reason):
    from agora_server.core.server.w2w_dataplane import build_push_response

    return build_push_response(status, reason)
