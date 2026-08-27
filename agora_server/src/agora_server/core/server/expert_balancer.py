"""Throughput-driven expert load balancer.

Selects the least-loaded expert of a stage via a min-heap of cumulative expected
runtime, estimated from a per-expert throughput EMA. Used by trainers to route
microbatches and by workers to pick their next hop in trainerless mode.
"""

import heapq
import random
import threading
import time

from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial

import grpc

from grpc._channel import _InactiveRpcError

from agora_server.core.server.dht_handler import ExpertInfo, extract_leaf_record, get_experts, parse_expert_dht_value
from agora_server.hivemind.dht import DHT, DHTNode, DHTValue
from agora_server.hivemind.moe.client.expert import RemoteExpert
from agora_server.hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from agora_server.hivemind.moe.expert_uid import ExpertPrefix, ExpertUID
from agora_server.hivemind.p2p import PeerID
from agora_server.hivemind.p2p.p2p_daemon_bindings.utils import P2PDaemonError
from agora_server.hivemind.utils import (
    DHTExpiration,
    PerformanceEMA,
    TimedStorage,
    ValueWithExpiration,
    get_dht_time,
    get_logger,
)
from agora_server.types import GhostPhase


logger = get_logger(__name__)


Endpoint = str


@dataclass(frozen=True)
class ExpertReservation:
    uid: ExpertUID
    peer_id: Endpoint
    task_size: float
    started_at: float


async def _get_uid_info(
    dht: DHT,
    node: DHTNode,
    uids: list[ExpertUID],
    expiration_time: DHTExpiration | None,
) -> list[tuple | None]:
    if expiration_time is None:
        expiration_time = get_dht_time()
    num_workers = len(uids) if dht.num_workers is None else min(len(uids), dht.num_workers)
    found: dict[ExpertUID, DHTValue] = await node.get_many(uids, expiration_time, num_workers=num_workers)

    experts: list[tuple | None] = [None] * len(uids)
    for i, uid in enumerate(uids):
        leaf = extract_leaf_record(found[uid])
        if leaf is not None and isinstance(leaf.value, str):
            peer_id, ghost_phase, _ = parse_expert_dht_value(leaf.value)
            experts[i] = (uid, peer_id.to_base58(), ghost_phase, leaf.expiration_time)
        else:
            experts[i] = (None, None, None, None)
    return experts


class ExpertBalancer:
    def __init__(
        self,
        dht: DHT,
        key: ExpertPrefix,
        update_period: float = 30.0,
        initial_throughput: float = 1.0,
        sleep_timeout: float = 5.0,
        ban_expiration: int = 3600,
        max_experts: int = None,
        min_experts: int = 1,
        refresh_experts_period: float = 600,
        trainer_rank: int = 0,
        warmup_batches: int = 0,
        warmup_penalty_multiplier: float = 1.0,
        start_update_thread_on_init: bool = True,
        **kwargs,
    ):
        self.dht, self.key = dht, key
        self.initial_throughput, self.ema_kwargs = initial_throughput, kwargs
        self.ban_expiration = ban_expiration
        self.max_experts = max_experts
        self.min_experts = min_experts
        self.refresh_experts_period = refresh_experts_period
        self.selected_experts = []
        self.trainer_rank = trainer_rank
        self.warmup_batches = warmup_batches
        self.warmup_penalty_multiplier = warmup_penalty_multiplier

        self.expert_verification_arguments = {}

        self.global_batch_counter = 0
        self.last_full_refresh = get_dht_time() - refresh_experts_period  # Force initial refresh
        self.experts = TimedStorage[ExpertUID, Endpoint]()
        self.blacklist = TimedStorage[ExpertUID, type(None)]()
        self.throughputs: dict[ExpertUID, PerformanceEMA] = {}
        # warmup table stores: uid -> (endpoint, global_batch_when_joined, last_runtime)
        # -1 for global_batch_when_joined means already warmed up
        self.warming_up: dict[ExpertUID, tuple[Endpoint, int, float]] = {}
        self.queue: list[tuple[float, float, ExpertUID]] = []
        self.uid_to_queue: dict[ExpertUID, tuple[float, float, ExpertUID]] = {}
        self.lock = threading.Lock()
        self.is_alive = threading.Event()
        self.is_alive.set()
        self.update_trigger, self.update_finished = threading.Event(), threading.Event()
        self.update_period, self.last_update = update_period, get_dht_time()
        self.sleep_timeout = sleep_timeout

        self._p2p = None

        self.update_thread = threading.Thread(target=self.update_experts_in_background, daemon=True)
        if start_update_thread_on_init:
            self.start_update_thread()

    def start_update_thread(self) -> None:
        self.update_thread.start()

    def update_experts_in_background(self):
        while self.is_alive.is_set():
            time_to_next_update = max(0.0, self.last_update + self.update_period - get_dht_time())
            try:
                self.update_trigger.wait(timeout=time_to_next_update)
            except TimeoutError:
                pass

            self.update_trigger.clear()

            time_since_full_refresh = get_dht_time() - self.last_full_refresh
            full_refresh_expire = time_since_full_refresh >= self.refresh_experts_period

            logger.info(f"Outside balancer check {self.key} num experts {len(self.experts)}")
            if self.selected_experts and not full_refresh_expire:
                # Subset refresh
                current_uids = [exp[0] for exp in self.selected_experts]
                expert_infos = self.dht.run_coroutine(partial(_get_uid_info, uids=current_uids, expiration_time=None))
                ghost1_count = 0
                ghost2_count = 0
                for i, uid in enumerate(current_uids):
                    _, peer_id, ghost_phase, expiration_time = expert_infos[i]

                    if peer_id is None:
                        self._evict_expert(uid)
                        continue

                    endpoint = peer_id

                    if ghost_phase == GhostPhase.PHASE1:
                        ghost1_count += 1
                        continue

                    if ghost_phase == GhostPhase.PHASE2:
                        ghost2_count += 1
                        # Phase 2: don't skip — add expert normally for trainer to send batches

                    # Check if expert endpoint changed and unban if different
                    maybe_banned = self._check_and_unban_if_endpoint_changed(uid, endpoint)

                    if maybe_banned is None or expiration_time > maybe_banned.expiration_time:
                        if not self._verify_expert(uid):
                            logger.warning(f"Expert {uid} with endpoint={endpoint} failed verification, banning.")
                            self.blacklist.store(uid, endpoint, get_dht_time() + self.ban_expiration)
                            logger.info(
                                f"Banned expert {uid} reason=verification_failure duration={self.ban_expiration}s"
                            )
                            continue
                        self._add_expert(uid, endpoint, expiration_time)
                    else:
                        logger.debug(f"Not adding expert {uid} (blacklisted).")
                logger.info(
                    f"Sub refresh {self.key} num experts {len(self.experts)}, "
                    f"skipped {ghost1_count} experts in ghost1 mode, {ghost2_count} experts in ghost2 mode."
                )

            # Decide if to do full refresh
            needs_full_refresh = (
                self.max_experts is None or full_refresh_expire or len(self.experts) < self.min_experts
            )

            if needs_full_refresh:
                # Full refresh
                response = self.dht.get(self.key, latest=True)

                if isinstance(response, ValueWithExpiration) and isinstance(response.value, dict):
                    all_experts = []
                    ghost1_count = 0
                    ghost2_count = 0
                    for _index, expert_info in response.value.items():
                        try:
                            (uid, value), expiration_time = expert_info
                            peer_id, ghost_phase, _ = parse_expert_dht_value(value)
                            endpoint = peer_id.to_base58()

                            if ghost_phase == GhostPhase.PHASE1:
                                ghost1_count += 1
                                continue

                            if ghost_phase == GhostPhase.PHASE2:
                                ghost2_count += 1
                                # Phase 2: don't skip — add expert normally for trainer to send batches

                            # Check if expert endpoint changed and unban if different
                            maybe_banned = self._check_and_unban_if_endpoint_changed(uid, endpoint)

                            if maybe_banned is None or expiration_time > maybe_banned.expiration_time:
                                if not self._verify_expert(uid):
                                    logger.warning(
                                        f"Expert {uid} with endpoint={endpoint} failed verification, banning."
                                    )
                                    self.blacklist.store(uid, endpoint, get_dht_time() + self.ban_expiration)
                                    logger.info(
                                        f"Banned expert {uid} reason=verification_failure "
                                        f"duration={self.ban_expiration}s"
                                    )
                                    continue
                                all_experts.append((uid, endpoint, expiration_time))
                            else:
                                logger.debug(f"Not adding expert {uid} (blacklisted).")

                        except Exception as e:
                            logger.warning(f"Skipping malformed expert info {expert_info} (exc={e})")

                    # If too many experts, select subset based on trainer_rank
                    logger.debug(f"Found {len(all_experts)} experts (max_experts: {self.max_experts})")
                    if self.max_experts and len(all_experts) > self.max_experts:
                        sorted_experts = sorted(all_experts, key=lambda x: int(x[0].split(".")[-1]))
                        offset = (self.trainer_rank * self.max_experts) % len(sorted_experts)

                        if offset + self.max_experts <= len(sorted_experts):
                            selected_experts = sorted_experts[offset : offset + self.max_experts]
                        else:
                            # Wrap around to beginning
                            selected_experts = (
                                sorted_experts[offset:]
                                + sorted_experts[: self.max_experts - (len(sorted_experts) - offset)]
                            )

                        self.selected_experts = selected_experts

                        # Clear current experts and add selected subset
                        # NOTE: warming_up is NOT cleared - it survives full refresh
                        with self.lock:
                            self.experts.clear()
                            self.uid_to_queue.clear()
                            self.queue.clear()

                        for uid, endpoint, expiration_time in selected_experts:
                            self._add_expert(uid, endpoint, expiration_time)
                    else:
                        # Add all experts (existing behavior when max_experts not exceeded)
                        for uid, endpoint, expiration_time in all_experts:
                            self._add_expert(uid, endpoint, expiration_time)

                    logger.info(
                        f"Full refresh {self.key} num experts {len(self.experts)}, time expiry {full_refresh_expire}, "
                        f"skipped {ghost1_count} experts in ghost1 mode, {ghost2_count} experts in ghost2 mode."
                    )
                    self.last_full_refresh = get_dht_time()
                else:
                    logger.warning(
                        f"Could not refresh experts, dht info key contains {response}, "
                        f"will retry in {time_to_next_update}s"
                    )

            if len(self.queue) == 0:
                logger.warning("Update routine finished, but still no experts available.")
                time.sleep(self.sleep_timeout)

            logger.info(f"Banned count {self.key} num banned {len(self.blacklist)}")

            self.last_update = get_dht_time()
            self.update_finished.set()

    def _ensure_single_uid_entry(self, uid: ExpertUID, new_endpoint: Endpoint):
        """Ensure that this UID has only one active entry in the warmup table.
        If the endpoint changed, this represents a new expert instance."""
        if uid in self.warming_up:
            stored_endpoint, _, _ = self.warming_up[uid]
            if stored_endpoint != new_endpoint:
                # Different endpoint = different expert instance
                logger.debug(
                    f"UID {uid} changed endpoint from {stored_endpoint} to {new_endpoint}. "
                    f"Removing old entry and starting fresh."
                )
                # Remove the old entry completely - it's a different physical expert
                del self.warming_up[uid]

    def _verify_expert(self, uid: ExpertUID) -> bool:
        """Verify that an expert's configuration matches expected values.
        Returns True if verification passes, False otherwise.
        """
        if not self.expert_verification_arguments:
            return True

        expert = get_experts(self.dht, [uid])[0]
        if expert is None:
            return False

        model_args = expert.info.get("model_args", {})
        for key, expected_value in self.expert_verification_arguments.items():
            if key not in model_args:
                logger.warning(f"Expert {uid} verification failed: missing required key '{key}'")
                return False
            if model_args[key] != expected_value:
                logger.warning(f"Expert {uid} verification failed: {key}={model_args[key]}, expected={expected_value}")
                return False

        return True

    def _add_expert(self, uid: ExpertUID, endpoint: Endpoint, expiration_time: DHTExpiration):
        with self.lock:
            self.experts.store(uid, endpoint, expiration_time)
            if uid not in self.uid_to_queue:
                logger.debug(f"Adding new expert: {uid}, expiration time = {expiration_time:.3f}.")

                # Ensure single entry per UID (removes old entry if endpoint changed)
                self._ensure_single_uid_entry(uid, endpoint)

                # Determine runtime/placement for this expert
                if uid in self.warming_up:
                    # Expert returning - use stored last_runtime
                    stored_endpoint, join_batch, last_runtime = self.warming_up[uid]
                    placement_runtime = last_runtime

                    if join_batch == -1:
                        # Already warmed up
                        logger.debug(f"Expert {uid} already warmed up, restoring to last_runtime={last_runtime:.3f}.")
                    else:
                        # Still warming up, keep existing warm-up progress
                        batches_elapsed = self.global_batch_counter - join_batch
                        logger.debug(
                            f"Expert {uid} keeping warm-up progress: {batches_elapsed}/{self.warmup_batches} "
                            f"global batches elapsed, restoring to last_runtime={last_runtime:.3f}."
                        )
                else:
                    # Brand new expert - calculate initial runtime and start warm-up
                    if len(self.queue) > 0:
                        placement_runtime = max(entry[0] for entry in self.queue)
                    else:
                        placement_runtime = 0.0

                    logger.debug(
                        f"Expert {uid} starting warm-up at global batch {self.global_batch_counter}, "
                        f"initial runtime={placement_runtime:.3f}."
                    )
                    self.warming_up[uid] = (endpoint, self.global_batch_counter, placement_runtime)

                # Initialize throughput tracking
                self.throughputs[uid] = PerformanceEMA(**self.ema_kwargs, paused=True)

                # Place expert in queue at its runtime position
                heap_entry = (placement_runtime, random.random(), uid)
                heapq.heappush(self.queue, heap_entry)
                self.uid_to_queue[uid] = heap_entry

                logger.debug(
                    f"Placed expert {uid} in queue with runtime={placement_runtime:.3f}. Queue size now: {len(self.queue)}"
                )
            else:
                logger.debug(f"Refreshing existing expert: {uid}, new expiration time = {expiration_time:.3f}.")

    def _evict_expert(self, uid: ExpertUID):
        """Evict an expert from the balancer."""
        with self.lock:
            if uid not in self.uid_to_queue:
                return
            self.uid_to_queue.pop(uid, None)
            self.throughputs.pop(uid, None)
            del self.experts[uid]
            logger.debug(f"Evicted dead expert {uid} (no longer present in DHT)")

    def _ban_expert(
        self, uid: ExpertUID, reason: str, ban_duration: float | None = None, only_if_present: bool = False
    ):
        effective_duration = ban_duration if ban_duration is not None else self.ban_expiration
        with self.lock:
            maybe_expert = self.experts.get(uid)
            if maybe_expert is None and only_if_present:
                # The expert is already gone (e.g. a prior ban inside use_specific_expert removed it
                # and stored a blacklist entry with its real endpoint).
                return
            if maybe_expert is None:
                expiration_time = get_dht_time()
                endpoint = None
            else:
                expiration_time = maybe_expert.expiration_time
                endpoint = maybe_expert.value
                # Keep warm-up state even when banned - it will survive if expert returns
                # with same endpoint during ban period
                del self.experts[uid]
            self.blacklist.store(uid, endpoint, get_dht_time() + effective_duration)
            self.uid_to_queue.pop(uid, None)
            self.throughputs.pop(uid, None)
            logger.debug(f"Banned expert {uid} with endpoint={endpoint}, expiration={expiration_time:.2f}")
        logger.info(f"Banned expert {uid} reason={reason} duration={effective_duration}s")

    def _is_expert_warming_up(self, uid: ExpertUID) -> bool:
        """Check if expert is currently warming up based on global batch counter."""
        if uid not in self.warming_up:
            return False

        _, join_batch, _ = self.warming_up[uid]
        if join_batch == -1:  # Sentinel: already warmed
            return False

        batches_elapsed = self.global_batch_counter - join_batch
        return batches_elapsed < self.warmup_batches

    def _update_last_runtime(self, uid: ExpertUID, new_runtime: float):
        """Update the last_runtime for an expert in the warmup table."""
        if uid in self.warming_up:
            endpoint, join_batch, _ = self.warming_up[uid]
            self.warming_up[uid] = (endpoint, join_batch, new_runtime)
            logger.debug(f"Updated last_runtime for expert {uid} to {new_runtime:.3f}.")

    def _check_and_unban_if_endpoint_changed(self, uid: ExpertUID, endpoint: Endpoint):
        """Check if expert endpoint changed and unban if different.

        When an expert rejoins with a different peer_id (e.g., after restart),
        remove it from blacklist to give it a fresh chance.

        Args:
            uid: Expert UID
            endpoint: Current peer_id for the expert

        Returns:
            Blacklist entry if expert is still banned, None if unbanned or not blacklisted
        """
        maybe_banned = self.blacklist.get(uid)
        if maybe_banned and maybe_banned.value != endpoint:
            logger.debug(f"Expert {uid} rejoined with different endpoint, removing from blacklist")
            del self.blacklist[uid]
            return None
        return maybe_banned

    def _expected_time_taken(self, uid: ExpertUID, task_size: float) -> float:
        if self.throughputs[uid].num_updates != 0:
            base_expected_time = task_size / self.throughputs[uid].samples_per_second
        else:
            base_expected_time = self.initial_throughput * task_size
        if self._is_expert_warming_up(uid):
            return base_expected_time * self.warmup_penalty_multiplier
        return base_expected_time

    def _reserve_queue_entry_locked(
        self, uid: ExpertUID, endpoint: Endpoint, current_runtime: float, task_size: float
    ) -> ExpertReservation:
        new_runtime = current_runtime + self._expected_time_taken(uid, task_size)
        new_heap_entry = (new_runtime, random.random(), uid)
        heapq.heappush(self.queue, new_heap_entry)
        self.uid_to_queue[uid] = new_heap_entry
        self._update_last_runtime(uid, new_runtime)
        return ExpertReservation(uid=uid, peer_id=endpoint, task_size=task_size, started_at=time.perf_counter())

    def reserve_another_expert(
        self, task_size: float, exclude_uids: set[ExpertUID] | None = None, block: bool = True
    ) -> ExpertReservation | None:
        """Reserve the least-loaded eligible expert, charging the heap at grant time.

        With ``block=False`` an empty stage returns ``None`` immediately (after nudging the
        membership refresh) instead of waiting for it - callers on a latency-sensitive path
        (the w2w local router) must drop rather than stall.
        """
        exclude_uids = set(exclude_uids or ())
        while True:
            if len(self.queue) == 0:
                if not block:
                    self.update_trigger.set()
                    return None
                self.update_finished.clear()
                self.update_trigger.set()
                self.update_finished.wait()
                continue

            skipped = []
            with self.lock:
                while self.queue:
                    current_runtime, _, uid = heap_entry = heapq.heappop(self.queue)
                    maybe_endpoint = self.experts.get(uid)

                    if maybe_endpoint is None:
                        self.uid_to_queue.pop(uid, None)
                        self.throughputs.pop(uid, None)
                        continue
                    if self.uid_to_queue.get(uid) != heap_entry:
                        continue
                    if uid in exclude_uids:
                        skipped.append(heap_entry)
                        continue

                    reservation = self._reserve_queue_entry_locked(
                        uid, maybe_endpoint.value, current_runtime, task_size
                    )
                    for skipped_entry in skipped:
                        heapq.heappush(self.queue, skipped_entry)
                    return reservation

                for skipped_entry in skipped:
                    heapq.heappush(self.queue, skipped_entry)
                if skipped:
                    return None

            if not block:
                self.update_trigger.set()
                return None
            self.update_finished.clear()
            self.update_trigger.set()
            self.update_finished.wait()

    def reserve_specific_expert(
        self, target_uid: ExpertUID, task_size: float, strict: bool = False
    ) -> ExpertReservation | None:
        with self.lock:
            maybe_endpoint = self.experts.get(target_uid)
            current_entry = self.uid_to_queue.get(target_uid)
            if maybe_endpoint is not None and current_entry is not None:
                return self._reserve_queue_entry_locked(target_uid, maybe_endpoint.value, current_entry[0], task_size)
        if strict:
            raise RuntimeError(f"pinned expert {target_uid} unavailable and strict pinning was requested")
        return None

    def settle_reservation(self, reservation: ExpertReservation, success: bool = True) -> None:
        if not success:
            return
        throughput = self.throughputs.get(reservation.uid)
        if throughput is None:
            return
        interval = max(time.perf_counter() - reservation.started_at, 1e-9)
        with throughput.lock:
            throughput.update(reservation.task_size, interval)
        self._mark_successful_batch(reservation.uid)

    def _mark_successful_batch(self, uid: ExpertUID) -> None:
        self.global_batch_counter += 1
        if uid in self.warming_up:
            endpoint, join_batch, last_runtime = self.warming_up[uid]
            if join_batch >= 0:
                batches_elapsed = self.global_batch_counter - join_batch
                if batches_elapsed >= self.warmup_batches:
                    self.warming_up[uid] = (endpoint, -1, last_runtime)
                    logger.info(f"Expert {uid} warm-up complete after {batches_elapsed} global batches!")
                else:
                    logger.debug(
                        f"Expert {uid} warm-up progress: {batches_elapsed}/{self.warmup_batches} global batches."
                    )

    @contextmanager
    def use_another_expert(self, task_size: float) -> RemoteExpert:
        while True:
            if len(self.queue) == 0:
                self.update_finished.clear()
                self.update_trigger.set()
                self.update_finished.wait()
                continue

            with self.lock:
                logger.debug(f"Getting a new expert, queue state: {self.queue}")
                current_runtime, _, uid = heap_entry = heapq.heappop(self.queue)
                maybe_endpoint = self.experts.get(uid)

                if maybe_endpoint is None:
                    # remove expired expert from queue
                    self.uid_to_queue.pop(uid, None)
                    self.throughputs.pop(uid, None)
                    # NOTE: We intentionally do NOT remove from warming_up here
                    # The expert might return quickly with the same endpoint
                if self.uid_to_queue.get(uid) != heap_entry:
                    logger.debug(
                        f"Skipping expert {uid} (uid_to_queue={self.uid_to_queue.get(uid)}, entry={heap_entry})"
                    )
                    continue  # skip uids that are banned or expired

                # Calculate base expected time
                if self.throughputs[uid].num_updates != 0:
                    base_expected_time = task_size / self.throughputs[uid].samples_per_second
                else:
                    base_expected_time = self.initial_throughput * task_size

                # Apply warm-up penalty if expert is warming up
                if self._is_expert_warming_up(uid):
                    expected_time_taken = base_expected_time * self.warmup_penalty_multiplier
                    _, join_batch, _ = self.warming_up[uid]
                    batches_elapsed = self.global_batch_counter - join_batch
                    logger.debug(
                        f"Expert {uid} is warming up ({batches_elapsed}/{self.warmup_batches} global batches elapsed), "
                        f"applying penalty multiplier {self.warmup_penalty_multiplier}x."
                    )
                else:
                    expected_time_taken = base_expected_time

                new_runtime = current_runtime + expected_time_taken
                new_heap_entry = (new_runtime, random.random(), uid)
                heapq.heappush(self.queue, new_heap_entry)
                self.uid_to_queue[uid] = new_heap_entry

                # Update the last_runtime in warmup table
                self._update_last_runtime(uid, new_runtime)

                break
        try:
            with self.throughputs[uid].update_threadsafe(task_size):
                logger.debug(f"Using expert {uid}, throughput = {self.throughputs[uid].samples_per_second}.")
                if self._p2p is None:
                    self._p2p = RemoteExpertWorker.run_coroutine(self.dht.replicate_p2p())
                expert_info = ExpertInfo(uid=uid, peer_id=PeerID.from_base58(maybe_endpoint.value))
                yield RemoteExpert(expert_info, self._p2p)
                logger.debug(f"Finished using expert {uid}.")

                # Increment global batch counter after successful completion
                self.global_batch_counter += 1

                # Check if expert just completed warm-up
                if uid in self.warming_up:
                    endpoint, join_batch, last_runtime = self.warming_up[uid]
                    if join_batch >= 0:  # Not already marked as warmed
                        batches_elapsed = self.global_batch_counter - join_batch
                        if batches_elapsed >= self.warmup_batches:
                            # Warm-up complete, set to -1 (sentinel for "already warmed")
                            self.warming_up[uid] = (endpoint, -1, last_runtime)
                            logger.info(f"Expert {uid} warm-up complete after {batches_elapsed} global batches!")
                        else:
                            logger.debug(
                                f"Expert {uid} warm-up progress: {batches_elapsed}/{self.warmup_batches} global batches."
                            )

        except _InactiveRpcError as error:
            if error.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                # response was too slow, choose the next expert
                raise
            else:
                # Network/connection error - ban expert
                logger.warning(f"Banning expert {uid} due to RPC error: {error.code()}")
                self._ban_expert(uid, reason="rpc_error")
                raise
        except P2PDaemonError as error:
            # Either the expert is unreachable or there is a network issue - short-ban the expert
            logger.warning(f"Short-banning expert {uid} for {self.update_period * 2}s due to P2P dial error: {error}")
            self._ban_expert(uid, reason="p2p_dial", ban_duration=self.update_period * 2)
            raise
        except (ConnectionError, TimeoutError, OSError) as error:
            # Network-related errors that indicate expert is unreachable
            logger.warning(f"Banning expert {uid} due to connection error: {type(error).__name__}: {error}")
            self._ban_expert(uid, reason="connection_error")
            raise
        except Exception as error:
            # Other exceptions (e.g., ValueError, TypeError, AttributeError) indicate bugs, not bad experts
            # Don't ban the expert, just propagate the error
            logger.error(f"Exception while using expert {uid} (not banning): {type(error).__name__}: {error}")
            raise

    @contextmanager
    def use_specific_expert(self, target_uid: ExpertUID, task_size: float, strict: bool = False) -> RemoteExpert:
        """Use a specific expert by UID for backward pass pinning.
        Falls back to heap-based selection if the target expert is unavailable (banned/expired),
        unless ``strict`` -- then a missing pin raises ``RuntimeError`` so the caller can drop the work
        instead of re-routing (worker-side activation caching: only the forward worker holds the cached
        activation)."""
        found_specific = False

        with self.lock:
            maybe_endpoint = self.experts.get(target_uid)
            current_entry = self.uid_to_queue.get(target_uid)

            if maybe_endpoint is not None and current_entry is not None:
                current_runtime = current_entry[0]

                if self.throughputs[target_uid].num_updates != 0:
                    base_expected_time = task_size / self.throughputs[target_uid].samples_per_second
                else:
                    base_expected_time = self.initial_throughput * task_size

                if self._is_expert_warming_up(target_uid):
                    expected_time_taken = base_expected_time * self.warmup_penalty_multiplier
                else:
                    expected_time_taken = base_expected_time

                new_runtime = current_runtime + expected_time_taken
                new_heap_entry = (new_runtime, random.random(), target_uid)
                heapq.heappush(self.queue, new_heap_entry)
                self.uid_to_queue[target_uid] = new_heap_entry
                self._update_last_runtime(target_uid, new_runtime)
                found_specific = True

        if not found_specific:
            if strict:
                raise RuntimeError(f"pinned expert {target_uid} unavailable and strict pinning was requested")
            logger.warning(f"Pinned expert {target_uid} unavailable, falling back to heap selection")
            with self.use_another_expert(task_size) as expert:
                yield expert
            return

        try:
            with self.throughputs[target_uid].update_threadsafe(task_size):
                logger.debug(
                    f"Using pinned expert {target_uid}, "
                    f"throughput = {self.throughputs[target_uid].samples_per_second}."
                )
                if self._p2p is None:
                    self._p2p = RemoteExpertWorker.run_coroutine(self.dht.replicate_p2p())
                expert_info = ExpertInfo(uid=target_uid, peer_id=PeerID.from_base58(maybe_endpoint.value))
                yield RemoteExpert(expert_info, self._p2p)
                logger.debug(f"Finished using pinned expert {target_uid}.")

                self.global_batch_counter += 1

                if target_uid in self.warming_up:
                    endpoint, join_batch, last_runtime = self.warming_up[target_uid]
                    if join_batch >= 0:
                        batches_elapsed = self.global_batch_counter - join_batch
                        if batches_elapsed >= self.warmup_batches:
                            self.warming_up[target_uid] = (endpoint, -1, last_runtime)
                            logger.info(
                                f"Expert {target_uid} warm-up complete after {batches_elapsed} global batches!"
                            )
                        else:
                            logger.debug(
                                f"Expert {target_uid} warm-up progress: "
                                f"{batches_elapsed}/{self.warmup_batches} global batches."
                            )

        except _InactiveRpcError as error:
            if error.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                raise
            else:
                logger.warning(f"Banning pinned expert {target_uid} due to RPC error: {error.code()}")
                self._ban_expert(target_uid, reason="rpc_error")
                raise
        except P2PDaemonError as error:
            logger.warning(
                f"Short-banning pinned expert {target_uid} for {self.update_period * 2}s "
                f"due to P2P dial error: {error}"
            )
            self._ban_expert(target_uid, reason="p2p_dial", ban_duration=self.update_period * 2)
            raise
        except (ConnectionError, TimeoutError, OSError) as error:
            logger.warning(
                f"Banning pinned expert {target_uid} due to connection error: {type(error).__name__}: {error}"
            )
            self._ban_expert(target_uid, reason="connection_error")
            raise
        except Exception as error:
            logger.error(
                f"Exception while using pinned expert {target_uid} (not banning): {type(error).__name__}: {error}"
            )
            raise

    def short_ban(self, uid: ExpertUID, reason: str = "w2w_pin_forward_failed") -> None:
        """Short-ban ``uid`` for ``update_period * 2`` (the same backoff used for P2P dial errors).

        Used by the w2w pin hook: when a pinned first forward attempt fails, the target is short-banned
        BEFORE the fallback retry so ``use_another_expert`` skips it. A present-but-slow/dead pinned
        worker -- e.g. a re-raised ``DEADLINE_EXCEEDED`` or a generic ``Exception`` that did not itself
        ban -- could otherwise stay queue-eligible and be re-selected, tight-looping the pipeline. The
        ban self-heals when it expires.

        ``only_if_present=True`` makes this precise: the bannable failures (P2P dial, non-deadline RPC,
        connection errors) are ALREADY banned inside ``use_specific_expert`` with the real endpoint, so
        here the expert is gone and ``short_ban`` is a deliberate no-op (re-banning would store a None
        endpoint and clobber that good entry). It bans only the non-banning failures -- exactly the
        cases that need it -- without coupling ``balanced_expert`` to the balancer's ban policy."""
        self._ban_expert(uid, reason=reason, ban_duration=self.update_period * 2, only_if_present=True)

    def snapshot_experts(self) -> list[tuple[ExpertUID, Endpoint]]:
        """Read-only view of the currently-known experts as ``[(uid, peer_id_base58)]``.

        Returns the experts a w2w route planner may pin a forward to: those present in
        ``self.experts`` (non-expired; ``TimedStorage.items()`` drops outdated entries) AND in
        ``self.uid_to_queue`` (queue-eligible, i.e. not banned/evicted) -- the same pair
        ``use_specific_expert`` requires before it pins instead of falling back. The stored value
        is the base58 ``peer_id`` string (cf. ``use_specific_expert``'s
        ``PeerID.from_base58(maybe_endpoint.value)``).

        Read-only: no heap, throughput, or warmup mutation -- it does not change selection behavior.
        """
        with self.lock:
            return [
                (uid, value_and_expiration.value)
                for uid, value_and_expiration in self.experts.items()
                if uid in self.uid_to_queue
            ]

    def peek_least_loaded(self) -> tuple[ExpertUID, Endpoint] | None:
        """Read-only: the queue-eligible expert with the least expected runtime, or ``None``.

        Returns the worker ``use_another_expert`` would pop NEXT -- the least-loaded valid expert
        (present in ``self.experts`` and queue-eligible in ``self.uid_to_queue``, the same validity
        ``snapshot_experts`` uses) by its current heap runtime -- as ``(uid, peer_id_base58)``. The w2w
        resolver uses this to pin/announce the load-aware target one stage ahead.

        Read-only: it does NOT pop or reserve (no heap/throughput mutation), so it never changes
        selection. Consequence: without a reservation, repeated peeks before any actual selection
        return the SAME worker -- in-flight microbatches can briefly herd until a real dispatch
        (``use_specific_expert`` / ``use_another_expert``) updates that worker's runtime.
        """
        with self.lock:
            best = None  # (runtime, uid, peer_id)
            for uid, value_and_expiration in self.experts.items():
                entry = self.uid_to_queue.get(uid)
                if entry is None:
                    continue
                runtime = entry[0]
                if best is None or runtime < best[0]:
                    best = (runtime, uid, value_and_expiration.value)
            if best is None:
                return None
            return (best[1], best[2])

    def shutdown(self):
        self.is_alive.clear()
        self.update_finished.clear()
        self.update_trigger.set()
        self.update_finished.wait()
