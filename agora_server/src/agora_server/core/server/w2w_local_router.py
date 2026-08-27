"""Sender-local next-hop routing for trainerless worker-to-worker mode.

Each worker selects the next hop for the microbatches it forwards with an
``ExpertBalancer``: the heap is charged when a hop is reserved, the throughput
EMA is settled when the receiver's processed-receipt arrives, failed hops are
short-banned (BUSY excepted), and a failure never rolls the heap back.

The send path runs inside forked ConnectionHandler processes:

- each process holds its own ``W2WLocalRouter`` (own heap/EMA/ban view);
- membership is fetched by one feeder thread in the main server process and
  shared through a manager dict; forked processes never call the store client;
- receipts land in a manager-dict mailbox stamped with ``time.monotonic()`` by
  the receiving process; the process owning the reservation settles the exact
  interval later. Both stamps come from the host-wide monotonic clock, so peer
  clocks are never compared.
"""

from __future__ import annotations

import threading
import time

from collections.abc import MutableMapping

from agora_server.core.server.dht_handler import parse_expert_dht_value
from agora_server.core.server.expert_balancer import ExpertBalancer, ExpertReservation
from agora_server.core.server.w2w_dataplane import (
    W2W_DROP,
    W2WHop,
    W2WReceiptServicer,
    W2WResolveResult,
)
from agora_server.hivemind.dht import DHT
from agora_server.hivemind.utils import ValueWithExpiration, get_logger
from agora_server.types import GhostPhase


logger = get_logger(__name__)

_MAILBOX_GC_PERIOD = 30.0


def stage_names(num_stages: int) -> list[str]:
    """Stage names in pipeline order: head, body1..bodyN, tail."""
    assert num_stages >= 2, "pipeline needs at least head and tail"
    return ["head"] + [f"body{i + 1}" for i in range(num_stages - 2)] + ["tail"]


def successor_stage_prefix(own_uid: str, num_stages: int) -> str | None:
    """Uid prefix of the next pipeline stage (``'body1.'``), or ``None`` for the tail."""
    own_stage = own_uid.split(".")[0]
    names = stage_names(num_stages)
    if own_stage not in names:
        raise ValueError(f"uid {own_uid!r} does not belong to a known stage of a {num_stages}-stage pipeline")
    index = names.index(own_stage)
    if index == len(names) - 1:
        return None
    return f"{names[index + 1]}."


class W2WMembership:
    """Successor-stage member snapshot shared across handler processes.

    ``shared`` is a manager dict (or a plain dict in single-process use). Members are
    ``(uid, peer_id_base58, expiration_time)`` tuples. The version bump happens after the
    member write so a reader observing a new version never sees older members.
    """

    def __init__(self, shared: MutableMapping):
        self._shared = shared

    def publish(self, members: list[tuple[str, str, float]]) -> None:
        self._shared["members"] = list(members)
        self._shared["version"] = int(self._shared.get("version", 0)) + 1

    def snapshot(self) -> tuple[int, list[tuple[str, str, float]]]:
        return int(self._shared.get("version", 0)), list(self._shared.get("members", ()))


class W2WMembershipFeeder(threading.Thread):
    """Main-process poller that publishes the successor stage's live experts.

    Runs only in the server's main process: forked handler processes must not call the
    store client, and the store then sees one membership poll per worker rather than one
    per handler process.
    """

    def __init__(self, *, dht: DHT, successor_prefix: str, membership: W2WMembership, update_period: float = 30.0):
        super().__init__(daemon=True, name=f"w2w-membership-{successor_prefix.rstrip('.')}")
        self._dht = dht
        self._key = f"{successor_prefix}0."
        self._membership = membership
        self._update_period = float(update_period)
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                self.refresh_once()
            except Exception as e:
                logger.warning(f"[W2WMembership] refresh of {self._key} failed: {type(e).__name__}: {e}")
            self._stop.wait(self._update_period)

    def refresh_once(self) -> None:
        members = fetch_stage_members(self._dht, key=self._key)
        if members is None:
            # Transient store hiccup: keep the last snapshot rather than wiping the stage.
            logger.warning(f"[W2WMembership] no member payload for {self._key}, keeping last snapshot")
            return
        self._membership.publish(members)
        logger.debug(f"[W2WMembership] {self._key}: {len(members)} members")

    def stop(self) -> None:
        self._stop.set()


def fetch_stage_members(dht: DHT, *, key: str) -> list[tuple[str, str, float]] | None:
    """Live ``(uid, peer_id, expiration)`` members under a stage key, ghost-phase-1 excluded.

    Returns ``None`` on a store hiccup so callers can keep their last snapshot.
    """
    response = dht.get(key, latest=True)
    if not (isinstance(response, ValueWithExpiration) and isinstance(response.value, dict)):
        return None
    members: list[tuple[str, str, float]] = []
    for expert_info in response.value.values():
        try:
            (uid, value), expiration_time = expert_info
            peer_id, ghost_phase, _ = parse_expert_dht_value(value)
            if ghost_phase == GhostPhase.PHASE1:
                continue
            members.append((uid, peer_id.to_base58(), float(expiration_time)))
        except Exception as e:
            logger.warning(f"[W2WMembership] skipping malformed expert info ({type(e).__name__}: {e})")
    return members


def join_order_data_rank(dht: DHT, own_uid: str) -> int:
    """Dense data-shard rank for a head: the number of other stage members visible at startup.

    Expert uids carry random suffixes, so uid order cannot provide dense ranks.
    Near-simultaneous joins can collide on a rank, which costs shard overlap for
    the run, not correctness.
    """
    prefix = own_uid.split(".")[0] + "."
    members = fetch_stage_members(dht, key=f"{prefix}0.") or []
    return sum(1 for uid, _, _ in members if uid != own_uid)


class W2WReceiptMailbox:
    """Cross-process receipt drop-box: (origin_uid, seq, receiver_uid) -> monotonic arrival time."""

    def __init__(self, shared: MutableMapping):
        self._shared = shared
        self._last_gc = time.monotonic()

    def post(self, *, trainer_uid: str, seq: int, uid: str) -> None:
        self._shared[(trainer_uid, int(seq), uid)] = time.monotonic()

    def take(self, key: tuple[str, int, str]) -> float | None:
        return self._shared.pop(key, None)

    def discard(self, key: tuple[str, int, str]) -> None:
        self._shared.pop(key, None)

    def gc(self, max_age: float) -> None:
        """Drop receipts nobody claimed (their lease failed or expired first)."""
        now = time.monotonic()
        if now - self._last_gc < _MAILBOX_GC_PERIOD:
            return
        self._last_gc = now
        for key, arrival in list(self._shared.items()):
            if now - arrival > max_age:
                self._shared.pop(key, None)


def make_receipt_servicer(mailbox: W2WReceiptMailbox) -> W2WReceiptServicer:
    """Receipt servicer that only stamps the mailbox; safe to register in any single process."""

    def post(*, trainer_uid: str, seq: int, uid: str) -> None:
        mailbox.post(trainer_uid=trainer_uid, seq=seq, uid=uid)

    return W2WReceiptServicer(post)


class W2WLocalRouter:
    """Per-process next-hop selector over the successor stage.

    ``resolve`` syncs membership from the shared snapshot, claims any receipts for its own
    reservations, then reserves the least-loaded successor on the local heap. A claimed
    receipt settles the reserve-to-receipt interval into that worker's throughput EMA.
    ``fail`` releases a reservation without a sample and short-bans the worker unless the
    failure was BUSY. Reservations with no receipt expire after ``lease_ttl`` seconds and
    are dropped without a sample (a lost receipt costs one sample, nothing else).
    """

    def __init__(
        self,
        *,
        successor_prefix: str,
        membership: W2WMembership,
        mailbox: W2WReceiptMailbox,
        lease_ttl: float,
        ban_expiration: int = 3600,
        **balancer_kwargs,
    ):
        self.successor_prefix = successor_prefix
        self._membership = membership
        self._mailbox = mailbox
        self._lease_ttl = float(lease_ttl)
        balancer_kwargs.setdefault("start_update_thread_on_init", False)
        self.balancer = ExpertBalancer(
            dht=None, key=f"{successor_prefix}0.", ban_expiration=ban_expiration, **balancer_kwargs
        )
        self._leases: dict[tuple[str, int, str], tuple[ExpertReservation, float, float]] = {}
        self._seen_version = -1
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}

    def resolve(
        self, *, origin_uid: str, seq: int, task_size: float, exclude_uids: set[str] | None = None
    ) -> W2WResolveResult:
        self._sync_membership()
        self._poll_receipts()
        reservation = self.balancer.reserve_another_expert(task_size, exclude_uids=exclude_uids, block=False)
        if reservation is None:
            self._bump("drops")
            return W2WResolveResult(status=W2W_DROP, reason="no eligible next worker")
        now = time.monotonic()
        with self._lock:
            self._leases[(origin_uid, int(seq), reservation.uid)] = (reservation, now, now + self._lease_ttl)
        self._bump("resolves")
        return W2WResolveResult(status="hop", hop=W2WHop(uid=reservation.uid, peer_id=reservation.peer_id))

    def fail(self, *, origin_uid: str, seq: int, uid: str, reason: str) -> None:
        key = (origin_uid, int(seq), uid)
        with self._lock:
            lease = self._leases.pop(key, None)
        self._mailbox.discard(key)
        if lease is not None:
            reservation, _, _ = lease
            self.balancer.settle_reservation(reservation, success=False)
        if reason != "busy":
            self.balancer.short_ban(uid, reason=f"w2w_{reason}")
        self._bump("failed")

    def stats(self) -> dict[str, int]:
        self._poll_receipts()
        with self._lock:
            stats = dict(self._counters)
            stats["open_leases"] = len(self._leases)
            return stats

    def _poll_receipts(self) -> None:
        now = time.monotonic()
        with self._lock:
            open_leases = list(self._leases.items())
        for key, (reservation, reserved_at, deadline) in open_leases:
            arrival = self._mailbox.take(key)
            if arrival is not None:
                with self._lock:
                    self._leases.pop(key, None)
                self._settle(reservation, max(arrival - reserved_at, 1e-9))
                self._bump("settled")
            elif deadline <= now:
                with self._lock:
                    self._leases.pop(key, None)
                self._bump("expired_leases")
        self._mailbox.gc(max_age=2 * self._lease_ttl)

    def _settle(self, reservation: ExpertReservation, interval: float) -> None:
        throughput = self.balancer.throughputs.get(reservation.uid)
        if throughput is None:
            return
        with throughput.lock:
            throughput.update(reservation.task_size, interval)
        self.balancer._mark_successful_batch(reservation.uid)

    def _sync_membership(self) -> None:
        version, members = self._membership.snapshot()
        if version == self._seen_version:
            return
        self._seen_version = version
        live = {uid for uid, _, _ in members}
        for uid in [uid for uid in list(self.balancer.uid_to_queue) if uid not in live]:
            self.balancer._evict_expert(uid)
        for uid, endpoint, expiration_time in members:
            maybe_banned = self.balancer._check_and_unban_if_endpoint_changed(uid, endpoint)
            if maybe_banned is None or expiration_time > maybe_banned.expiration_time:
                self.balancer._add_expert(uid, endpoint, expiration_time)

    def _bump(self, name: str, delta: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + delta
