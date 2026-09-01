# Copyright 2026 Pluralis Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Direct worker-to-worker data path helpers.

The protocol is deliberately small: workers ask the trainer coordinator for the next hop, report
accepted/progress/loss/done/drop events, and respond to peer pushes with ACCEPTED/BUSY/DROP.
Tensor transport still reuses the existing expert RPC service; direct mode is selected by metadata
and by wiring this driver into ``ConnectionHandler``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import multiprocessing as mp
import time

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch

from agora_server.core.address_book import transport_maddrs
from agora_server.core.batch_metadata import BatchMetadata
from agora_server.hivemind.compression import serialize_torch_tensor
from agora_server.hivemind.p2p import P2P, P2PContext, PeerID, ServicerBase
from agora_server.hivemind.p2p.p2p_daemon_bindings.control import DEFAULT_MAX_MSG_SIZE, MAX_UNARY_PAYLOAD_SIZE
from agora_server.hivemind.proto import runtime_pb2
from agora_server.hivemind.utils import get_logger, nested_flatten
from agora_server.hivemind.utils.asyncio import amap_in_executor, iter_as_aiter
from agora_server.hivemind.utils.serializer import MSGPackSerializer
from agora_server.hivemind.utils.streaming import split_for_streaming


logger = get_logger(__name__)

W2W_COORD_NAMESPACE = "w2w_coord"
W2W_RECEIPT_NAMESPACE = "w2w_receipt"
W2W_TERMINUS = "terminus"
W2W_DROP = "drop"
W2W_HOP_LOG_EVERY = 50


class W2WPushStatus(IntEnum):
    ACCEPTED = 0
    BUSY = 1
    DROP = 2


class W2WReportKind(IntEnum):
    ACCEPTED = 0
    LOSS = 1
    BWD_PROGRESS = 2
    DONE = 3
    DROPPED = 4


@dataclass(frozen=True)
class W2WHop:
    uid: str
    peer_id: str
    maddrs: tuple[str, ...] = ()


@dataclass(frozen=True)
class W2WResolveResult:
    status: str
    hop: W2WHop | None = None
    reason: str = ""


def _pack(obj: dict[str, Any]) -> bytes:
    return MSGPackSerializer.dumps(obj)


def _unpack(raw: bytes) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        obj = MSGPackSerializer.loads(raw)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def build_push_response(status: W2WPushStatus, reason: str = "") -> runtime_pb2.ExpertResponse:
    return runtime_pb2.ExpertResponse(metadata=_pack({"status": int(status), "reason": reason}))


def parse_push_response(response: runtime_pb2.ExpertResponse) -> tuple[W2WPushStatus, str]:
    obj = _unpack(response.metadata)
    try:
        status = W2WPushStatus(int(obj.get("status", int(W2WPushStatus.ACCEPTED))))
    except Exception:
        status = W2WPushStatus.DROP
    reason = obj.get("reason", "")
    return status, reason if isinstance(reason, str) else ""


def build_resolve_request(
    *,
    trainer_uid: str,
    seq: int,
    current_uid: str,
    task_size: float,
    failed_uid: str | None = None,
    failed_reason: str | None = None,
) -> runtime_pb2.ExpertRequest:
    return runtime_pb2.ExpertRequest(
        uid=trainer_uid,
        metadata=_pack(
            {
                "trainer_uid": trainer_uid,
                "seq": int(seq),
                "current_uid": current_uid,
                "task_size": float(task_size),
                "failed_uid": failed_uid,
                "failed_reason": failed_reason,
            }
        ),
    )


def build_resolve_response(result: W2WResolveResult) -> runtime_pb2.ExpertResponse:
    obj: dict[str, Any] = {"status": result.status, "reason": result.reason}
    if result.hop is not None:
        obj.update(uid=result.hop.uid, peer_id=result.hop.peer_id, maddrs=list(result.hop.maddrs))
    return runtime_pb2.ExpertResponse(metadata=_pack(obj))


def parse_resolve_response(response: runtime_pb2.ExpertResponse) -> W2WResolveResult:
    obj = _unpack(response.metadata)
    status = obj.get("status", W2W_DROP)
    if status == "hop":
        uid = obj.get("uid")
        peer_id = obj.get("peer_id")
        maddrs = obj.get("maddrs", ())
        if isinstance(uid, str) and isinstance(peer_id, str):
            if not (isinstance(maddrs, (list, tuple)) and all(isinstance(addr, str) for addr in maddrs)):
                maddrs = ()
            return W2WResolveResult(status="hop", hop=W2WHop(uid, peer_id, tuple(maddrs)))
    if status == W2W_TERMINUS:
        return W2WResolveResult(status=W2W_TERMINUS, reason=str(obj.get("reason", "")))
    return W2WResolveResult(status=W2W_DROP, reason=str(obj.get("reason", "")))


def build_report_request(
    *,
    trainer_uid: str,
    seq: int,
    kind: W2WReportKind,
    uid: str,
    prev_uid: str | None = None,
    loss: float | None = None,
    reason: str | None = None,
) -> runtime_pb2.ExpertRequest:
    return runtime_pb2.ExpertRequest(
        uid=trainer_uid,
        metadata=_pack(
            {
                "trainer_uid": trainer_uid,
                "seq": int(seq),
                "kind": int(kind),
                "uid": uid,
                "prev_uid": prev_uid,
                "loss": loss,
                "reason": reason,
            }
        ),
    )


def build_receipt_request(*, trainer_uid: str, seq: int, uid: str) -> runtime_pb2.ExpertRequest:
    return runtime_pb2.ExpertRequest(
        uid=uid, metadata=_pack({"trainer_uid": trainer_uid, "seq": int(seq), "uid": uid})
    )


class W2WReceiptServicer(ServicerBase):
    """Receives processed-receipts from next-hop workers and settles the sender's reservations.

    Trainerless mode only: the receipt confirms the receiver finished its forward compute, which
    is what ends the sender-side interval feeding the throughput EMA.
    """

    def __init__(self, settle_fn: Callable[..., None]):
        self._settle_fn = settle_fn

    async def rpc_w2w_processed(
        self, request: runtime_pb2.ExpertRequest, context: P2PContext
    ) -> runtime_pb2.ExpertResponse:
        obj = _unpack(request.metadata)
        self._settle_fn(
            trainer_uid=str(obj.get("trainer_uid", "")),
            seq=int(obj.get("seq", -1)),
            uid=str(obj.get("uid", request.uid)),
        )
        return build_push_response(W2WPushStatus.ACCEPTED)


class W2WCoordServicer(ServicerBase):
    def __init__(
        self,
        resolve_fn: Callable[..., W2WResolveResult],
        report_fn: Callable[..., None],
    ):
        self._resolve_fn = resolve_fn
        self._report_fn = report_fn

    async def rpc_resolve_next_hop(
        self, request: runtime_pb2.ExpertRequest, context: P2PContext
    ) -> runtime_pb2.ExpertResponse:
        obj = _unpack(request.metadata)
        result = self._resolve_fn(
            trainer_uid=str(obj.get("trainer_uid", request.uid)),
            seq=int(obj.get("seq", -1)),
            current_uid=str(obj.get("current_uid", "")),
            task_size=float(obj.get("task_size", 1.0)),
            failed_uid=obj.get("failed_uid") if isinstance(obj.get("failed_uid"), str) else None,
            failed_reason=obj.get("failed_reason") if isinstance(obj.get("failed_reason"), str) else None,
        )
        return build_resolve_response(result)

    async def rpc_report_event(
        self, request: runtime_pb2.ExpertRequest, context: P2PContext
    ) -> runtime_pb2.ExpertResponse:
        obj = _unpack(request.metadata)
        try:
            kind = W2WReportKind(int(obj.get("kind", int(W2WReportKind.DROPPED))))
        except Exception:
            kind = W2WReportKind.DROPPED
        self._report_fn(
            trainer_uid=str(obj.get("trainer_uid", request.uid)),
            seq=int(obj.get("seq", -1)),
            kind=kind,
            uid=str(obj.get("uid", "")),
            prev_uid=obj.get("prev_uid") if isinstance(obj.get("prev_uid", None), str) else None,
            loss=obj.get("loss") if isinstance(obj.get("loss"), (int, float)) else None,
            reason=obj.get("reason") if isinstance(obj.get("reason"), str) else None,
        )
        return build_push_response(W2WPushStatus.ACCEPTED)


class W2WCompletionSlot:
    def __init__(self, trainer_uid: str, seq: int):
        self.trainer_uid = trainer_uid
        self.seq = seq
        self.loss: float | None = None
        self.done = False
        self.dropped = False
        self.drop_reason = ""
        self.last_progress = time.monotonic()
        self.accepted: list[tuple[str, str | None]] = []
        self._lock = mp.Lock()
        self._event = mp.Event()

    def report(
        self,
        kind: W2WReportKind,
        *,
        uid: str = "",
        prev_uid: str | None = None,
        loss: float | None = None,
        reason: str | None = None,
    ) -> None:
        with self._lock:
            self.last_progress = time.monotonic()
            if kind == W2WReportKind.ACCEPTED:
                self.accepted.append((uid, prev_uid))
            elif kind == W2WReportKind.LOSS:
                self.loss = float(loss) if loss is not None else None
            elif kind == W2WReportKind.DONE:
                self.done = True
                self._event.set()
            elif kind == W2WReportKind.DROPPED:
                self.dropped = True
                self.drop_reason = reason or "worker reported drop"
                self._event.set()

    def wait(self, timeout: float, stall_timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if self._event.wait(min(remaining, 1.0)):
                return True
            if time.monotonic() - self.last_progress > stall_timeout:
                with self._lock:
                    self.dropped = True
                    self.drop_reason = f"no w2w progress for {stall_timeout:.1f}s"
                    self._event.set()
                return True


@dataclass(frozen=True)
class _AcceptedForward:
    metadata: BatchMetadata
    key: tuple[str, str, int, str]
    is_new: bool


class DirectW2WDriver:
    def __init__(
        self,
        *,
        max_entries: int,
        forward_timeout: float,
        backward_timeout: float,
        coord_timeout: float = 30.0,
        manager=None,
        name: str = "",
        trainerless: bool = False,
        local_router=None,
        origin_ledger=None,
        own_peer_id: str | None = None,
        receipt_mailbox=None,
    ):
        if max_entries <= 0:
            raise ValueError(f"max_entries must be positive, got {max_entries}")
        if local_router is not None and not trainerless:
            raise ValueError("local_router requires trainerless=True")
        if origin_ledger is not None and own_peer_id is None:
            raise ValueError("origin_ledger requires own_peer_id to detect loopback reports")
        self.max_entries = int(max_entries)
        self.forward_timeout = float(forward_timeout)
        self.backward_timeout = float(backward_timeout)
        self.coord_timeout = float(coord_timeout)
        self.name = name
        # Trainerless: hops are resolved by the local router (None on the tail, which never
        # resolves) and every finished forward compute is receipted back to the sender.
        self.trainerless = bool(trainerless)
        self._local_router = local_router
        self.origin_ledger = origin_ledger
        self._own_peer_id = own_peer_id
        # Shared with ConnectionHandler processes, which register the receipt servicer on it.
        self.receipt_mailbox = receipt_mailbox
        self._owns_manager = manager is None
        self._manager = manager if manager is not None else mp.Manager()
        self._entries = self._manager.dict()
        self._prev_hops = self._manager.dict()
        self._counters = self._manager.dict()
        self._lock = self._manager.Lock()
        # Strong refs to in-flight fire-and-forget progress-report tasks (per handler process, so a
        # plain set -- asyncio tasks never cross processes). Without this asyncio keeps only a weak ref
        # and could GC an in-flight report mid-send.
        self._report_tasks: set = set()

    def _spawn_report(
        self, p2p: P2P, md: BatchMetadata, kind: W2WReportKind, *, uid: str, prev_uid: str | None = None
    ) -> None:
        """Send a bookkeeping report (ACCEPTED / BWD_PROGRESS) without blocking the push ACK, retaining a
        strong ref so asyncio does not GC the in-flight task. Result-bearing reports (LOSS / DONE /
        DROPPED) are NOT routed here -- they stay synchronous so the trainer's completion slot observes
        the loss and terminal state in order."""
        task = asyncio.ensure_future(self._report_safe(p2p, md, kind, uid=uid, prev_uid=prev_uid))
        self._report_tasks.add(task)
        task.add_done_callback(self._report_tasks.discard)

    async def _report_safe(
        self, p2p: P2P, md: BatchMetadata, kind: W2WReportKind, *, uid: str, prev_uid: str | None = None
    ) -> None:
        try:
            await self.report_event(p2p, md, kind, uid=uid, prev_uid=prev_uid)
        except Exception as e:
            self._bump("report_errors")
            logger.debug(
                f"[W2WReport] async {kind.name} report failed uid={uid} seq={md.seq}: {type(e).__name__}: {e}"
            )

    def _spawn_receipt(self, p2p: P2P, md: BatchMetadata, *, receiver_uid: str) -> None:
        """Send a processed-receipt to the sender of this microbatch, off the critical path.

        Fire-and-forget: a lost receipt costs the sender one EMA sample (its lease expires),
        nothing else. No-op outside trainerless mode or when there is no previous hop (the
        head's own stage-0 compute is local and unreserved)."""
        if not self.trainerless or not md.prev_hop_peer_id:
            return
        task = asyncio.ensure_future(self._send_receipt_safe(p2p, md, receiver_uid=receiver_uid))
        self._report_tasks.add(task)
        task.add_done_callback(self._report_tasks.discard)

    async def _send_receipt_safe(self, p2p: P2P, md: BatchMetadata, *, receiver_uid: str) -> None:
        try:
            stub = W2WReceiptServicer.get_stub(
                p2p, PeerID.from_base58(md.prev_hop_peer_id), namespace=W2W_RECEIPT_NAMESPACE
            )
            await stub.rpc_w2w_processed(
                build_receipt_request(trainer_uid=md.trainer_uid, seq=int(md.seq), uid=receiver_uid),
                timeout=self.coord_timeout,
            )
        except Exception as e:
            self._bump("receipt_errors")
            logger.debug(f"[W2WReceipt] send failed seq={md.seq} -> {md.prev_hop_uid}: {type(e).__name__}: {e}")

    async def accept_forward(
        self, uid: str, metadata: bytes, p2p: P2P
    ) -> tuple[W2WPushStatus, str, _AcceptedForward | None]:
        md = BatchMetadata.from_bytes(metadata)
        if md is None or md.trainer_peer_id is None:
            self._bump("drops")
            return W2WPushStatus.DROP, "missing direct w2w metadata", None
        key = (uid, md.trainer_uid, int(md.seq), "f")
        with self._lock:
            if key in self._entries:
                self._bump_locked("duplicates")
                return W2WPushStatus.ACCEPTED, "duplicate", _AcceptedForward(md, key, False)
            if len(self._entries) >= self.max_entries:
                self._bump_locked("busy")
                return W2WPushStatus.BUSY, "ingress full", None
            self._entries[key] = time.monotonic()
            self._bump_locked("accepted_forward")

        # TODO: _prev_hops is only popped by the backward; also evict on eval / forward-drop / ack-fail paths or entries leak (follow-up PR).
        if md.prev_hop_uid and md.prev_hop_peer_id:
            self._prev_hops[(uid, md.trainer_uid, int(md.seq))] = (
                md.prev_hop_uid,
                md.prev_hop_peer_id,
                tuple(md.prev_hop_maddrs or ()),
            )

        # Fire the ACCEPTED report off the critical path: it only settles the previous hop's lease and
        # bumps the trainer's stall heartbeat (carries no result), so the sender's push ACK must not wait
        # on a trainer round-trip. Completion is gated by the synchronous LOSS/DONE; a failed or
        # post-teardown report is harmless (the latter is dropped by the trainer as a late_report).
        self._spawn_report(p2p, md, W2WReportKind.ACCEPTED, uid=uid, prev_uid=md.prev_hop_uid)
        return W2WPushStatus.ACCEPTED, "", _AcceptedForward(md, key, True)

    async def accept_backward(
        self, uid: str, metadata: bytes, p2p: P2P
    ) -> tuple[W2WPushStatus, str, _AcceptedForward | None]:
        md = BatchMetadata.from_bytes(metadata)
        if md is None or md.trainer_peer_id is None:
            self._bump("drops")
            return W2WPushStatus.DROP, "missing direct w2w metadata", None
        key = (uid, md.trainer_uid, int(md.seq), "b")
        with self._lock:
            if key in self._entries:
                self._bump_locked("duplicates")
                return W2WPushStatus.ACCEPTED, "duplicate", _AcceptedForward(md, key, False)
            if len(self._entries) >= self.max_entries:
                self._bump_locked("busy")
                return W2WPushStatus.BUSY, "ingress full", None
            self._entries[key] = time.monotonic()
            self._bump_locked("accepted_backward")
        # Fire BWD_PROGRESS off the critical path (stall heartbeat / bookkeeping only) -- see accept_forward.
        self._spawn_report(p2p, md, W2WReportKind.BWD_PROGRESS, uid=uid, prev_uid=md.prev_hop_uid)
        return W2WPushStatus.ACCEPTED, "", _AcceptedForward(md, key, True)

    async def handle_forward(
        self,
        *,
        uid: str,
        accepted: _AcceptedForward,
        forward_inputs: Sequence[torch.Tensor],
        passthrough: Sequence[torch.Tensor],
        passthrough_serialized: Sequence[runtime_pb2.Tensor],
        expert: Any,
        p2p: P2P,
        this_peer_id: str,
        this_maddrs: Sequence[str],
    ) -> None:
        md = accepted.metadata
        # The fused tail consumes (hidden, labels, loss_weight) and pushes a gradient backward instead
        # of forwarding, so it never reserves a next hop; every other stage does. The reservation depends
        # only on task_size -- the batch dimension (tensors[0].shape[0]), identical on the input and the
        # output -- not on the forward result. So kick the next-hop reservation RPC off concurrently with
        # the forward compute: this overlaps the trainer round-trip with GPU time instead of stacking it
        # serially after the compute (that serial round-trip was the dominant per-microbatch latency the
        # direct path added over the relay path).
        is_tail_stage = len(forward_inputs) >= 3 and not passthrough
        resolve_future = None
        try:
            if not is_tail_stage:
                resolve_future = asyncio.ensure_future(
                    self.resolve_next_hop(p2p, md, current_uid=uid, task_size=_task_size(forward_inputs))
                )
            raw_outputs = tuple(await expert.forward_pool.submit_task(*forward_inputs))
            # Trainerless: confirm the compute to the sender so it can settle its reservation.
            # Fired after submit_task so the receipt covers transfer + queue + compute.
            self._spawn_receipt(p2p, md, receiver_uid=uid)
            if len(raw_outputs) == 2 and len(forward_inputs) >= 3 and not passthrough:
                await self._handle_tail_forward(uid, md, raw_outputs, forward_inputs, expert, p2p)
                return

            output_serialized = [
                serialize_torch_tensor(result, proto.compression, allow_inplace=True)
                for result, proto in zip(raw_outputs, nested_flatten(expert.outputs_schema))
            ]
            tensors = list(output_serialized) + list(passthrough_serialized)
            size_tensors = list(raw_outputs) + list(passthrough)
            next_md = dataclasses.replace(
                md,
                next_hop_uid=None,
                next_hop_peer_id=None,
                prev_hop_uid=uid,
                prev_hop_peer_id=this_peer_id,
                prev_hop_maddrs=tuple(this_maddrs),
            )
            ok = await self._push_forward_with_retries(
                p2p, md, uid, next_md, tensors, size_tensors, prefetched_hop=resolve_future
            )
            resolve_future = None  # consumed (awaited) inside _push_forward_with_retries
            if not ok:
                await self.report_event(p2p, md, W2WReportKind.DROPPED, uid=uid, reason="forward push failed")
        except Exception as e:
            # Cancel a still-pending next-hop reservation BEFORE announcing the terminal drop, so a
            # resolve in flight does not reserve a worker for a microbatch we have already abandoned.
            # (The trainer's slot_active guard + drop_slot cleanup also catch a resolve whose RPC already
            # left the wire; cancelling here just avoids the wasted reservation and keeps the drop path
            # self-evidently correct rather than relying on that teardown.)
            if resolve_future is not None:
                resolve_future.cancel()
                resolve_future = None
            logger.warning(f"[W2WSend] forward DROP uid={uid} seq={md.seq}: {type(e).__name__}: {e}")
            await self._safe_report_drop(p2p, md, uid, f"forward failed: {type(e).__name__}")
        finally:
            # Backstop: cancel the prefetch on any path that did not consume it (e.g. an early return).
            if resolve_future is not None:
                resolve_future.cancel()
            self.release(accepted.key)

    async def handle_backward(
        self,
        *,
        uid: str,
        accepted: _AcceptedForward,
        inputs_and_grads: Sequence[torch.Tensor],
        expert: Any,
        p2p: P2P,
    ) -> None:
        md = accepted.metadata
        try:
            grad_inputs = tuple(await expert.backward_pool.submit_task(*inputs_and_grads))
            prev = self._prev_hops.pop((uid, md.trainer_uid, int(md.seq)), None)
            if prev is None:
                await self.report_event(p2p, md, W2WReportKind.DONE, uid=uid)
                self._bump("backward_done")
                return
            prev_uid, prev_peer_id, prev_maddrs = prev
            serialized = [
                serialize_torch_tensor(result, proto.compression, allow_inplace=True)
                for result, proto in zip(grad_inputs, nested_flatten(expert.grad_inputs_schema))
            ]
            ok = await self._push_backward(
                p2p, md, W2WHop(prev_uid, prev_peer_id, tuple(prev_maddrs)), serialized, grad_inputs
            )
            if not ok:
                await self.report_event(p2p, md, W2WReportKind.DROPPED, uid=uid, reason="backward push failed")
        except Exception as e:
            logger.warning(f"[W2WSend] backward DROP uid={uid} seq={md.seq}: {type(e).__name__}: {e}")
            await self._safe_report_drop(p2p, md, uid, f"backward failed: {type(e).__name__}")
        finally:
            self.release(accepted.key)

    async def _handle_tail_forward(
        self,
        uid: str,
        md: BatchMetadata,
        raw_outputs: Sequence[torch.Tensor],
        forward_inputs: Sequence[torch.Tensor],
        expert: Any,
        p2p: P2P,
    ) -> None:
        loss_per_token, grad_hidden = raw_outputs
        loss = float(loss_per_token.detach().float().mean().item())
        await self.report_event(p2p, md, W2WReportKind.LOSS, uid=uid, loss=loss)
        is_train = len(forward_inputs) >= 3 and bool(torch.any(forward_inputs[2]))
        prev = self._prev_hops.pop((uid, md.trainer_uid, int(md.seq)), None)
        if not is_train or prev is None:
            await self.report_event(p2p, md, W2WReportKind.DONE, uid=uid)
            return
        prev_uid, prev_peer_id, prev_maddrs = prev
        grad_hidden_cpu = grad_hidden.cpu().detach()
        hidden_proto = tuple(nested_flatten(expert.forward_schema))[0]
        serialized = [serialize_torch_tensor(grad_hidden_cpu, hidden_proto.compression, allow_inplace=True)]
        ok = await self._push_backward(
            p2p, md, W2WHop(prev_uid, prev_peer_id, tuple(prev_maddrs)), serialized, (grad_hidden_cpu,)
        )
        if not ok:
            await self.report_event(p2p, md, W2WReportKind.DROPPED, uid=uid, reason="tail backward push failed")

    async def _push_forward_with_retries(
        self,
        p2p: P2P,
        md: BatchMetadata,
        current_uid: str,
        next_md: BatchMetadata,
        tensors: Sequence[runtime_pb2.Tensor],
        size_tensors: Sequence[torch.Tensor],
        prefetched_hop: asyncio.Future | None = None,
    ) -> bool:
        failed_uid = md.next_hop_uid
        failed_reason = None
        # Per-batch exclusion for the trainerless local resolver; each resolve here is stateless.
        # The trainer path ignores it (the coordinator tracks exclusions server-side from failed_uid).
        excluded: set[str] = {failed_uid} if failed_uid else set()
        deadline = time.monotonic() + self.forward_timeout
        try:
            while time.monotonic() < deadline:
                if prefetched_hop is not None:
                    # First attempt reuses the reservation started concurrently with the forward compute
                    # (no failed_uid yet -- it is the first try). Retries below resolve fresh.
                    hop = await prefetched_hop
                    prefetched_hop = None
                else:
                    hop = await self.resolve_next_hop(
                        p2p,
                        md,
                        current_uid=current_uid,
                        task_size=_task_size(size_tensors),
                        failed_uid=failed_uid,
                        failed_reason=failed_reason,
                        exclude_uids=excluded,
                    )
                    failed_uid = failed_reason = None
                if hop.status != "hop" or hop.hop is None:
                    return False
                try:
                    await _connect_if_needed(p2p, hop.hop, self.coord_timeout)
                    stub = _get_server_stub(p2p, hop.hop.peer_id)
                    status, reason = await _call_push_rpc(
                        stub, "forward", hop.hop.uid, tensors, next_md.to_bytes(), size_tensors, self.forward_timeout
                    )
                    if status == W2WPushStatus.ACCEPTED:
                        self._bump("forward_pushes")
                        logger.debug(f"[W2WSend] forward uid={current_uid} seq={md.seq} -> {hop.hop.uid} accepted")
                        return True
                    # BUSY is transient backpressure, not a bad worker: keep its reason as the no-ban sentinel.
                    failed_reason = "busy" if status == W2WPushStatus.BUSY else (reason or status.name.lower())
                    failed_uid = hop.hop.uid
                    excluded.add(hop.hop.uid)
                    self._bump("busy_retries")
                except Exception as e:
                    failed_uid, failed_reason = hop.hop.uid, type(e).__name__
                    excluded.add(hop.hop.uid)
                    with self._lock:
                        self._bump_locked("push_errors")
                        self._bump_locked(f"push_fail:{hop.hop.uid}")
                    logger.warning(
                        f"[W2WSend] forward push uid={current_uid} seq={md.seq} -> {hop.hop.uid} "
                        f"failed: {type(e).__name__}: {e}"
                    )
            logger.warning(
                f"[W2WSend] forward push uid={current_uid} seq={md.seq} gave up after "
                f"{self.forward_timeout:.0f}s: excluded {sorted(excluded)}, last failure: {failed_reason}"
            )
            return False
        finally:
            if prefetched_hop is not None:
                prefetched_hop.cancel()
            # Failures normally ride the next resolve; one on the final attempt has no
            # next resolve, so its lease is settled and the worker banned here.
            if failed_uid is not None and self.trainerless and self._local_router is not None:
                self._local_router.fail(
                    origin_uid=md.trainer_uid, seq=int(md.seq), uid=failed_uid, reason=failed_reason or "push_failed"
                )

    async def _push_backward(
        self,
        p2p: P2P,
        md: BatchMetadata,
        hop: W2WHop,
        tensors: Sequence[runtime_pb2.Tensor],
        size_tensors: Sequence[torch.Tensor],
    ) -> bool:
        bwd_md = dataclasses.replace(
            md,
            next_hop_uid=None,
            next_hop_peer_id=None,
            prev_hop_uid=None,
            prev_hop_peer_id=None,
            prev_hop_maddrs=None,
        )
        try:
            await _connect_if_needed(p2p, hop, self.coord_timeout)
            stub = _get_server_stub(p2p, hop.peer_id)
            status, reason = await _call_push_rpc(
                stub, "backward", hop.uid, tensors, bwd_md.to_bytes(), size_tensors, self.backward_timeout
            )
            if status == W2WPushStatus.ACCEPTED:
                self._bump("backward_pushes")
                if int(md.seq) % W2W_HOP_LOG_EVERY == 0:
                    logger.info(f"[W2WHop] backward sent: {self.name} -> {hop.uid}@{hop.peer_id}")
                return True
            logger.warning(f"[W2WSend] backward push to {hop.uid} rejected: {status.name} {reason}")
        except Exception as e:
            logger.warning(f"[W2WSend] backward push to {hop.uid} failed: {type(e).__name__}: {e}")
            with self._lock:
                self._bump_locked("push_errors")
                self._bump_locked(f"push_fail:{hop.uid}")
        return False

    async def resolve_next_hop(
        self,
        p2p: P2P,
        md: BatchMetadata,
        *,
        current_uid: str,
        task_size: float,
        failed_uid: str | None = None,
        failed_reason: str | None = None,
        exclude_uids: set[str] | None = None,
    ) -> W2WResolveResult:
        if self.trainerless:
            router = self._local_router
            if failed_uid is not None and router is not None:
                router.fail(
                    origin_uid=md.trainer_uid, seq=int(md.seq), uid=failed_uid, reason=failed_reason or "push_failed"
                )
            if router is None:
                return W2WResolveResult(status=W2W_TERMINUS)
            return router.resolve(
                origin_uid=md.trainer_uid, seq=int(md.seq), task_size=task_size, exclude_uids=exclude_uids
            )
        stub = W2WCoordServicer.get_stub(p2p, PeerID.from_base58(md.trainer_peer_id), namespace=W2W_COORD_NAMESPACE)
        response = await stub.rpc_resolve_next_hop(
            build_resolve_request(
                trainer_uid=md.trainer_uid,
                seq=md.seq,
                current_uid=current_uid,
                task_size=task_size,
                failed_uid=failed_uid,
                failed_reason=failed_reason,
            ),
            timeout=self.coord_timeout,
        )
        return parse_resolve_response(response)

    async def report_event(
        self,
        p2p: P2P,
        md: BatchMetadata,
        kind: W2WReportKind,
        *,
        uid: str,
        prev_uid: str | None = None,
        loss: float | None = None,
        reason: str | None = None,
    ) -> None:
        # A head worker reporting on its own batch would dial itself; hand the report to the
        # in-process ledger instead (the ledger state is manager-backed, so this also works
        # from forked handler processes).
        if self.origin_ledger is not None and md.trainer_peer_id == self._own_peer_id:
            self.origin_ledger.report_event(
                trainer_uid=md.trainer_uid,
                seq=int(md.seq),
                kind=kind,
                uid=uid,
                prev_uid=prev_uid,
                loss=loss,
                reason=reason,
            )
            return
        stub = W2WCoordServicer.get_stub(p2p, PeerID.from_base58(md.trainer_peer_id), namespace=W2W_COORD_NAMESPACE)
        await stub.rpc_report_event(
            build_report_request(
                trainer_uid=md.trainer_uid,
                seq=md.seq,
                kind=kind,
                uid=uid,
                prev_uid=prev_uid,
                loss=loss,
                reason=reason,
            ),
            timeout=self.coord_timeout,
        )

    async def _safe_report_drop(self, p2p: P2P, md: BatchMetadata, uid: str, reason: str) -> None:
        try:
            await self.report_event(p2p, md, W2WReportKind.DROPPED, uid=uid, reason=reason)
        except Exception:
            pass

    def release(self, key: tuple[str, str, int, str]) -> None:
        with self._lock:
            self._entries.pop(key, None)

    def stats(self) -> dict[str, int]:
        with self._lock:
            stats = {str(k): int(v) for k, v in self._counters.items()}
            stats["entries"] = len(self._entries)
            return stats

    def close(self) -> None:
        if self._owns_manager:
            self._manager.shutdown()

    def _bump(self, name: str, delta: int = 1) -> None:
        with self._lock:
            self._bump_locked(name, delta)

    def _bump_locked(self, name: str, delta: int = 1) -> None:
        self._counters[name] = int(self._counters.get(name, 0)) + delta


def _task_size(tensors: Sequence[torch.Tensor]) -> float:
    if tensors:
        try:
            return float(tensors[0].shape[0])
        except Exception:
            pass
    return 1.0


def _serialized_nbytes(tensors: Iterable[runtime_pb2.Tensor]) -> int:
    return sum(len(t.buffer) for t in tensors)


async def _call_push_rpc(
    stub,
    direction: str,
    uid: str,
    tensors: Sequence[runtime_pb2.Tensor],
    metadata: bytes,
    size_tensors: Sequence[torch.Tensor],
    timeout: float,
) -> tuple[W2WPushStatus, str]:
    method = stub.rpc_forward if direction == "forward" else stub.rpc_backward
    stream_method = stub.rpc_forward_stream if direction == "forward" else stub.rpc_backward_stream
    if _serialized_nbytes(tensors) <= MAX_UNARY_PAYLOAD_SIZE:
        response = await method(
            runtime_pb2.ExpertRequest(uid=uid, tensors=list(tensors), metadata=metadata), timeout=timeout
        )
        return parse_push_response(response)

    split = [part for tensor in tensors for part in split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE)]

    async def _stream_push():
        responses = await stream_method(
            amap_in_executor(
                lambda pair: runtime_pb2.ExpertRequest(
                    uid=uid, tensors=[pair[1]], metadata=(metadata if pair[0] == 0 else b"")
                ),
                iter_as_aiter(enumerate(split)),
            )
        )
        async for response in responses:
            return parse_push_response(response)
        return W2WPushStatus.DROP, "empty stream response"

    # Bound the streaming push (large activations): a hung peer must not stall the coroutine forever
    # (the unary branch above is already bounded by `timeout=`); on timeout the caller drops + retries.
    return await asyncio.wait_for(_stream_push(), timeout)


def _get_server_stub(p2p: P2P, peer_id: str):
    from agora_server.hivemind.moe.client.expert import get_server_stub

    return get_server_stub(p2p, PeerID.from_base58(peer_id))


async def _connect_if_needed(p2p: P2P, hop: W2WHop, timeout: float | None = None) -> None:
    if not hop.maddrs:
        return
    # transport_maddrs strips the /p2p/<id> suffix (the go p2pd daemon rejects it) and validates the embedded id
    peer = PeerID.from_base58(hop.peer_id)
    transport = transport_maddrs(list(hop.maddrs), peer)
    if not transport:
        return
    # bound the dial so an unreachable peer can't hang the push (caller's retry loop drops it on timeout)
    await asyncio.wait_for(p2p._client.connect(peer, transport), timeout)
