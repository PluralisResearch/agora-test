"""Batch origination on head workers in trainerless worker-to-worker mode.

A head worker replaces the trainer for the batches it originates: a background
thread reads pretokenized Pithos samples into a bounded prefetch queue, a
credit-gated loop injects microbatches into the worker's own forward path (the
same ``DirectW2WDriver`` accept/handle used for remote pushes, so ingress
accounting, routing, receipts, and drop reporting are identical), and a
manager-dict ledger collects the LOSS/DONE/DROPPED reports that downstream
workers already send to ``BatchMetadata.trainer_peer_id`` - now the head's own
peer id. Completed batches feed the same ``TrainerMetricsReporter`` records the
loss monitor scrapes from trainers.

Ledger state is a manager dict with one key per field: reports arrive from the
injector's own process and from forked ConnectionHandler processes, and only
single-key writes are atomic across processes.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import math
import queue
import threading
import time

from collections.abc import Callable, Generator, Iterator, MutableMapping
from dataclasses import dataclass

import torch

from tenacity import (
    RetryCallState,
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_delay,
    wait_exponential,
)
from tenacity.stop import stop_base

from agora_server.core.batch_metadata import BatchMetadata
from agora_server.core.metrics_reporter import TrainerMetricsReporter
from agora_server.core.server.w2w_dataplane import (
    W2W_DROP,
    DirectW2WDriver,
    W2WCoordServicer,
    W2WPushStatus,
    W2WReportKind,
    W2WResolveResult,
)
from agora_server.hivemind.compression import serialize_torch_tensor
from agora_server.hivemind.p2p import P2P
from agora_server.hivemind.proto import runtime_pb2
from agora_server.hivemind.utils import get_logger
from agora_server.types import GhostPhase
from pithos import CacheConfig, Corpus
from pithos.errors import DownloadError
from pithos.registry import resolve_corpus
from pithos.torch import PithosBatchSource


logger = get_logger(__name__)

_WAIT_POLL_S = 0.05
_SUMMARY_EVERY = 200
_DROP_BACKOFF_S = 1.0
_REAP_TICK_S = 0.05
_ORPHAN_SWEEP_TICKS = 200
_GHOST_PHASE_POLL_S = 0.5
_DOWNLOAD_RETRY_BUDGET_S = 600.0
_DOWNLOAD_RETRY_MAX_WAIT_S = 60.0


class _StopWhenSet(stop_base):
    """Tenacity stop condition that fires as soon as `event` is set."""

    def __init__(self, event: threading.Event) -> None:
        self._event = event

    def __call__(self, retry_state: RetryCallState) -> bool:
        return self._event.is_set()


def _download_retrying(abort: threading.Event) -> Retrying:
    """The retry policy for pithos transfers, cancellable through `abort`.

    Only DownloadError is retried: it is pithos's "transfer failed" class (chunk and
    manifest fetches alike), and its journaled partials make a retry resume rather than
    restart. Identity, integrity, and config errors stay immediately terminal. Setting
    `abort` wakes the backoff sleep and stops further attempts, so shutdown is never
    held behind the retry budget.
    """

    def sleep_until_abort(seconds: float) -> None:
        abort.wait(seconds)

    return Retrying(
        retry=retry_if_exception_type(DownloadError),
        wait=wait_exponential(multiplier=1, max=_DOWNLOAD_RETRY_MAX_WAIT_S),
        stop=stop_after_delay(_DOWNLOAD_RETRY_BUDGET_S) | _StopWhenSet(abort),
        sleep=sleep_until_abort,
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


@dataclass(frozen=True)
class PithosBatch:
    """One token batch and its immutable corpus sample range."""

    input_ids: torch.Tensor
    manifest_identity: str
    sample_start: int
    sample_stride: int


class _GuardedBatchIterator(Iterator[PithosBatch]):
    def __init__(self, iterator: Generator[PithosBatch, None, None], release: Callable[[], None]) -> None:
        self._iterator = iterator
        self._release = release
        self._closed = False

    def __iter__(self) -> _GuardedBatchIterator:
        return self

    def __next__(self) -> PithosBatch:
        return next(self._iterator)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._iterator.close()
        self._release()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


def _resolve_corpus_location(
    corpus_name: str | None,
    corpus_uri: str | None,
    expected_manifest_identity: str,
    cache_budget_bytes: int,
    prefetch_depth: int,
    registry_path: str | None,
) -> str:
    """The corpus URI to open: an explicit ``corpus_uri`` wins, else the
    registry entry's — the packaged registry may omit the location entirely
    (the authorizer supplies it on authed runs), while its identity and
    reader-override pins are still enforced whenever a name is given.
    """
    if corpus_uri is not None and (not isinstance(corpus_uri, str) or not corpus_uri):
        raise ValueError(f"corpus_uri must be a nonempty str, got {corpus_uri!r}")
    if corpus_name is None:
        if corpus_uri is None:
            raise ValueError("a corpus_name and/or a corpus_uri is required")
        return corpus_uri
    entry = resolve_corpus(corpus_name, registry_path)
    if entry.manifest_identity != expected_manifest_identity:
        raise ValueError(
            f"configured manifest identity {expected_manifest_identity!r} does not match "
            f"registry identity {entry.manifest_identity!r} for {corpus_name!r}"
        )
    registry_budget = getattr(entry, "budget_bytes", None)
    if registry_budget is not None and registry_budget != cache_budget_bytes:
        raise ValueError(
            f"configured cache budget {cache_budget_bytes} does not match registry budget {registry_budget}"
        )
    registry_prefetch = getattr(entry, "prefetch_depth", None)
    if registry_prefetch is not None and registry_prefetch != prefetch_depth:
        raise ValueError(
            f"configured prefetch depth {prefetch_depth} does not match registry prefetch depth {registry_prefetch}"
        )
    if corpus_uri is not None:
        return corpus_uri
    if entry.uri is None:
        raise ValueError(
            f"corpus {corpus_name!r} has no uri in the registry and no pithos_corpus_uri was given; "
            "the authorizer supplies it on authed runs — pass pithos_corpus_uri=<url> for manual runs"
        )
    return entry.uri


class PithosBatchStream:
    """Stateful iterator factory for one deterministic Pithos stream."""

    def __init__(
        self,
        *,
        corpus_name: str,
        corpus_uri: str | None = None,
        expected_manifest_identity: str,
        cache_dir: str,
        cache_budget_bytes: int,
        prefetch_depth: int,
        sequence_length: int,
        seed: int,
        stream_count: int,
        data_idx: int,
        batch_size: int,
        registry_path: str | None = None,
    ) -> None:
        positive_values = (
            ("cache_budget_bytes", cache_budget_bytes),
            ("sequence_length", sequence_length),
            ("stream_count", stream_count),
            ("batch_size", batch_size),
        )
        for name, value in positive_values:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if isinstance(prefetch_depth, bool) or not isinstance(prefetch_depth, int) or prefetch_depth < 0:
            raise ValueError(f"prefetch_depth must be a nonnegative int, got {prefetch_depth!r}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed must be an int, got {seed!r}")
        if isinstance(data_idx, bool) or not isinstance(data_idx, int) or not 0 <= data_idx < stream_count:
            raise ValueError(f"data_idx {data_idx!r} must be in [0, {stream_count})")
        self.corpus_name = corpus_name
        self._corpus_uri = _resolve_corpus_location(
            corpus_name, corpus_uri, expected_manifest_identity, cache_budget_bytes, prefetch_depth, registry_path
        )
        self.manifest_identity = expected_manifest_identity
        self.cache_dir = cache_dir
        self.cache_budget_bytes = cache_budget_bytes
        self.prefetch_depth = prefetch_depth
        self.sequence_length = sequence_length
        self.seed = seed
        self.stream_count = stream_count
        self.data_idx = data_idx
        self.batch_size = batch_size
        self.registry_path = registry_path
        self._next_stream_offset = 0
        self._iterator_active = False
        self._abort = threading.Event()
        self._lock = threading.Lock()

    def __call__(self) -> _GuardedBatchIterator:
        """Open the corpus and yield batches until the caller stops iteration."""
        with self._lock:
            if self._iterator_active:
                raise RuntimeError("PithosBatchStream supports one active iterator")
            self._iterator_active = True
        return _GuardedBatchIterator(self._iterate_batches(), self._release_iterator)

    def abort(self) -> None:
        """Cancel in-progress download retries: wake the backoff sleep and fail the read."""
        self._abort.set()

    def _release_iterator(self) -> None:
        with self._lock:
            self._iterator_active = False

    def _open_corpus(self) -> Corpus:
        return Corpus.from_uri(
            self._corpus_uri,
            self.sequence_length,
            self.seed,
            CacheConfig(
                cache_dir=self.cache_dir,
                budget_bytes=self.cache_budget_bytes,
                prefetch_depth=self.prefetch_depth,
            ),
            expected_manifest_identity=self.manifest_identity,
        )

    def _iterate_batches(self) -> Generator[PithosBatch, None, None]:
        try:
            retrying = _download_retrying(self._abort)
            corpus = retrying(self._open_corpus)
            with PithosBatchSource(corpus) as source:
                if source.identity != self.manifest_identity:
                    raise ValueError(
                        f"opened manifest identity {source.identity!r} does not match {self.manifest_identity!r}"
                    )
                logger.info(
                    f"[W2WInject] verified Pithos stream {self.data_idx}/{self.stream_count}, "
                    f"manifest {self.manifest_identity}"
                )
                while True:
                    with self._lock:
                        stream_offset = self._next_stream_offset
                    sample_start = self.data_idx + stream_offset * self.stream_count
                    input_ids = retrying(source.read_strided, sample_start, self.stream_count, self.batch_size)
                    with self._lock:
                        if self._next_stream_offset != stream_offset:
                            raise RuntimeError("PithosBatchStream position changed during a read")
                        self._next_stream_offset += self.batch_size
                    yield PithosBatch(
                        input_ids=input_ids,
                        manifest_identity=self.manifest_identity,
                        sample_start=sample_start,
                        sample_stride=self.stream_count,
                    )
        finally:
            self._release_iterator()


class PithosShardStream:
    """Stateful iterator factory for one explicitly assigned corpus shard.

    A shard is one corpus chunk file: ``shard_count`` must equal the corpus
    chunk count, and this head reads only chunks ``[shard_index *
    shards_per_head, (shard_index + 1) * shards_per_head)``, restarting from
    its first chunk each epoch — pithos reshuffles within chunks per epoch.
    The shard index and the corpus location come from the authorizer for
    contributor-run heads, or from explicit config for manual runs; the
    registry ``corpus_name`` pins identity and reader overrides while
    ``corpus_uri`` supplies the location (see ``_resolve_corpus_location``).
    """

    def __init__(
        self,
        *,
        corpus_name: str | None = None,
        corpus_uri: str | None = None,
        expected_manifest_identity: str,
        cache_dir: str,
        cache_budget_bytes: int,
        prefetch_depth: int,
        sequence_length: int,
        seed: int,
        shard_index: int,
        shard_count: int,
        shards_per_head: int,
        batch_size: int,
        registry_path: str | None = None,
    ) -> None:
        positive_values = (
            ("cache_budget_bytes", cache_budget_bytes),
            ("sequence_length", sequence_length),
            ("shard_count", shard_count),
            ("shards_per_head", shards_per_head),
            ("batch_size", batch_size),
        )
        for name, value in positive_values:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if isinstance(prefetch_depth, bool) or not isinstance(prefetch_depth, int) or prefetch_depth < 0:
            raise ValueError(f"prefetch_depth must be a nonnegative int, got {prefetch_depth!r}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed must be an int, got {seed!r}")
        if isinstance(shard_index, bool) or not isinstance(shard_index, int) or shard_index < 0:
            raise ValueError(f"shard_index must be a nonnegative int, got {shard_index!r}")
        if (shard_index + 1) * shards_per_head > shard_count:
            raise ValueError(
                f"shard_index {shard_index} with shards_per_head {shards_per_head} exceeds shard_count {shard_count}"
            )
        self.corpus_name = corpus_name
        self._corpus_uri = _resolve_corpus_location(
            corpus_name, corpus_uri, expected_manifest_identity, cache_budget_bytes, prefetch_depth, registry_path
        )
        self.manifest_identity = expected_manifest_identity
        self.cache_dir = cache_dir
        self.cache_budget_bytes = cache_budget_bytes
        self.prefetch_depth = prefetch_depth
        self.sequence_length = sequence_length
        self.seed = seed
        self.shard_index = shard_index
        self.shard_count = shard_count
        self.shards_per_head = shards_per_head
        self.batch_size = batch_size
        self.registry_path = registry_path
        self._epoch = 0
        self._next_shard_offset = 0
        self._iterator_active = False
        self._abort = threading.Event()
        self._lock = threading.Lock()

    def __call__(self) -> _GuardedBatchIterator:
        """Open the corpus and yield batches until the caller stops iteration."""
        with self._lock:
            if self._iterator_active:
                raise RuntimeError("PithosShardStream supports one active iterator")
            self._iterator_active = True
        return _GuardedBatchIterator(self._iterate_batches(), self._release_iterator)

    def abort(self) -> None:
        """Cancel in-progress download retries: wake the backoff sleep and fail the read."""
        self._abort.set()

    def _release_iterator(self) -> None:
        with self._lock:
            self._iterator_active = False

    def _open_corpus(self) -> Corpus:
        return Corpus.from_uri(
            self._corpus_uri,
            self.sequence_length,
            self.seed,
            CacheConfig(
                cache_dir=self.cache_dir,
                budget_bytes=self.cache_budget_bytes,
                prefetch_depth=self.prefetch_depth,
            ),
            expected_manifest_identity=self.manifest_identity,
        )

    def _iterate_batches(self) -> Generator[PithosBatch, None, None]:
        try:
            retrying = _download_retrying(self._abort)
            corpus = retrying(self._open_corpus)
            with PithosBatchSource(corpus) as source:
                if source.identity != self.manifest_identity:
                    raise ValueError(
                        f"opened manifest identity {source.identity!r} does not match {self.manifest_identity!r}"
                    )
                total = source.total_samples
                if self.shard_count != source.chunk_count:
                    raise ValueError(
                        f"shard_count {self.shard_count} does not match corpus chunk count "
                        f"{source.chunk_count} — shard mode assigns one shard per chunk file"
                    )
                samples_per_chunk = source.samples_per_chunk
                shard_lo = self.shard_index * self.shards_per_head * samples_per_chunk
                shard_hi = min((self.shard_index + 1) * self.shards_per_head * samples_per_chunk, total)
                shard_size = shard_hi - shard_lo
                if shard_size < 1:
                    raise ValueError(
                        f"shard {self.shard_index} holds no samples at this sequence length "
                        f"(samples [{shard_lo}, {shard_hi}) of {total})"
                    )
                logger.info(
                    f"[W2WInject] verified Pithos shard {self.shard_index}/{self.shard_count} "
                    f"(x{self.shards_per_head}): samples [{shard_lo}, {shard_hi}) of {total}, "
                    f"manifest {self.manifest_identity}"
                )
                while True:
                    with self._lock:
                        epoch, offset = self._epoch, self._next_shard_offset
                    # Never read across the shard boundary: the last batch of an
                    # epoch may be short (the injector derives sizes from shape).
                    rows = min(self.batch_size, shard_size - offset)
                    sample_start = epoch * total + shard_lo + offset
                    input_ids = retrying(source.read, sample_start, rows)
                    with self._lock:
                        if (self._epoch, self._next_shard_offset) != (epoch, offset):
                            raise RuntimeError("PithosShardStream position changed during a read")
                        if offset + rows >= shard_size:
                            self._epoch += 1
                            self._next_shard_offset = 0
                        else:
                            self._next_shard_offset = offset + rows
                    yield PithosBatch(
                        input_ids=input_ids,
                        manifest_identity=self.manifest_identity,
                        sample_start=sample_start,
                        sample_stride=1,
                    )
        finally:
            self._release_iterator()


class W2WOriginLedger:
    """Per-origin batch completion state, shared across processes via a manager dict.

    Keys are ``(origin_uid, seq, field)`` with fields ``progress`` (monotonic stamp),
    ``loss``, ``done``, ``dropped``, ``reason``. Reports for unknown batches (already
    timed out and dropped) are counted and ignored.
    """

    def __init__(self, shared: MutableMapping, origin_uid: str):
        self._shared = shared
        self._origin_uid = origin_uid
        self._counters: dict[str, int] = {}
        self._lock = threading.Lock()
        self._servicer = W2WCoordServicer(self._resolve_stub, self.report_event)
        self._p2ps: list[P2P] = []

    @property
    def servicer(self) -> W2WCoordServicer:
        """Report servicer, registered by the connection handlers (see ConnectionHandler.run)."""
        return self._servicer

    def new_slot(self, seq: int) -> None:
        self._shared[(self._origin_uid, int(seq), "progress")] = time.monotonic()

    def drop_slot(self, seq: int) -> None:
        for field in ("progress", "loss", "done", "dropped", "reason"):
            self._shared.pop((self._origin_uid, int(seq), field), None)

    def report_event(
        self,
        *,
        trainer_uid: str,
        seq: int,
        kind: W2WReportKind,
        uid: str,
        prev_uid: str | None = None,
        loss: float | None = None,
        reason: str | None = None,
    ) -> None:
        key = (trainer_uid, int(seq), "progress")
        if key not in self._shared:
            self._bump("late_reports")
            return
        self._shared[key] = time.monotonic()
        if kind == W2WReportKind.LOSS and loss is not None:
            self._shared[(trainer_uid, int(seq), "loss")] = float(loss)
        elif kind == W2WReportKind.DONE:
            self._shared[(trainer_uid, int(seq), "done")] = True
        elif kind == W2WReportKind.DROPPED:
            self._shared[(trainer_uid, int(seq), "reason")] = reason or "worker reported drop"
            self._shared[(trainer_uid, int(seq), "dropped")] = True
        self._bump(kind.name.lower())

    def wait(self, seq: int, *, timeout: float, stall_timeout: float) -> tuple[str, float | None, str]:
        """Block until the batch completes; returns (outcome, loss, reason).

        Outcome is ``"done"``, ``"dropped"``, or ``"timeout"``; a stall (no report for
        ``stall_timeout``) counts as dropped, mirroring the trainer's completion slot.
        """
        deadline = time.monotonic() + timeout
        base = (self._origin_uid, int(seq))
        while time.monotonic() < deadline:
            if self._shared.get((*base, "dropped")):
                return "dropped", None, str(self._shared.get((*base, "reason"), ""))
            if self._shared.get((*base, "done")):
                loss = self._shared.get((*base, "loss"))
                return "done", (float(loss) if loss is not None else None), ""
            progress = self._shared.get((*base, "progress"))
            if progress is not None and time.monotonic() - progress > stall_timeout:
                return "dropped", None, f"no w2w progress for {stall_timeout:.1f}s"
            time.sleep(_WAIT_POLL_S)
        return "timeout", None, f"no completion after {timeout:.1f}s"

    def stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def snapshot(self) -> dict:
        """Copy every slot field in one atomic manager round-trip.

        Must be ``copy()``: ``dict(proxy)`` fetches keys and values in separate
        round-trips, so a concurrent drop_slot raises KeyError mid-copy.
        """
        return self._shared.copy()

    def _resolve_stub(self, **_kwargs) -> W2WResolveResult:
        return W2WResolveResult(status=W2W_DROP, reason="trainerless origin does not resolve")

    def _bump(self, name: str, delta: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + delta


class _AdaptiveCredits:
    """Occupancy-band in-flight window for the injector.

    A round is ``target`` completions. After each round, the estimated number of
    this head's batches queued downstream is

        occupancy = target * (1 - best_median / round_median)

    The window grows by one below [occupancy_low, occupancy_high], shrinks by one
    above it, holds inside it. A congestion signal (drop, timeout, rejected
    injection) halves the window, with a cooldown so one burst counts once.
    Heads sharing a bottleneck aim for the same queued count, so equal windows
    are the equilibrium; a head re-grows whenever its queue share falls.
    """

    def __init__(
        self,
        *,
        initial: int = 1,
        ceiling: int = 64,
        cooldown_s: float = 10.0,
        occupancy_low: float = 4.0,
        occupancy_high: float = 10.0,
    ):
        self.target = max(1, int(initial))
        self._ceiling = max(1, int(ceiling))
        self._cooldown_s = float(cooldown_s)
        self._occ_low = float(occupancy_low)
        self._occ_high = float(occupancy_high)
        self._round_done = 0
        self._round_latencies: list[float] = []
        self._best_median: float | None = None
        self._last_backoff = 0.0
        self.last_occupancy = 0.0

    def on_done(self, latency_s: float) -> None:
        self._round_done += 1
        self._round_latencies.append(latency_s)
        if self._round_done < self.target:
            return
        ordered = sorted(self._round_latencies)
        median = ordered[len(ordered) // 2]
        if self._best_median is None or median < self._best_median:
            self._best_median = median
        occupancy = self.target * max(0.0, 1.0 - self._best_median / median) if median > 0 else 0.0
        self.last_occupancy = occupancy
        if occupancy < self._occ_low and self.target < self._ceiling:
            self.target += 1
        elif occupancy > self._occ_high and self.target > 1:
            self.target -= 1
        self._round_done = 0
        self._round_latencies.clear()

    def on_congestion(self) -> None:
        now = time.monotonic()
        if now - self._last_backoff < self._cooldown_s:
            return
        self._last_backoff = now
        self.target = max(1, self.target // 2)
        self._round_done = 0
        self._round_latencies.clear()


class W2WHeadInjector:
    """Credit-gated batch origination loop for one head worker.

    Injects each microbatch through the worker's own ``accept_forward`` +
    ``handle_forward``, waits on the ledger until the batch's reverse chain
    finishes, and feeds the reporter. ``max_inflight`` credits bound the number
    of unfinished batches. Ghost phase 1 pauses new admissions while accepted
    batches drain; phase 2 and normal operation admit batches.
    """

    def __init__(
        self,
        *,
        driver: DirectW2WDriver,
        ledger: W2WOriginLedger,
        reporter: TrainerMetricsReporter,
        dht,
        expert_uid: str,
        expert_backend,
        batch_source: Callable[[], Iterator[PithosBatch | torch.Tensor]],
        num_stages: int,
        data_idx: int,
        max_inflight: int = 1,
        prefetch_batches: int = 4,
        forward_timeout: float = 225.0,
        backward_timeout: float = 562.5,
        activation_cache=None,
        adaptive_inflight: bool = False,
        occupancy_low: float = 4.0,
        occupancy_high: float = 10.0,
        data_start_timeout: float = 900.0,
    ):
        assert max_inflight >= 1
        if (
            isinstance(data_start_timeout, bool)
            or not isinstance(data_start_timeout, (int, float))
            or not math.isfinite(data_start_timeout)
            or data_start_timeout <= 0
        ):
            raise ValueError(f"data_start_timeout must be positive, got {data_start_timeout!r}")
        self._driver = driver
        self._ledger = ledger
        self._reporter = reporter
        self._dht = dht
        self._expert_uid = expert_uid
        self._expert = expert_backend
        if expert_backend.optimizer is None:
            raise ValueError("trainerless head injector requires a training coordinator")
        self._training_coordinator = expert_backend.optimizer
        self._activation_cache = activation_cache
        self._batch_source = batch_source
        self._manifest_identity = getattr(batch_source, "manifest_identity", None)
        self._data_idx = int(data_idx)
        self._max_inflight = int(max_inflight)
        # Adaptive mode: max_inflight becomes the safety ceiling for the adaptive window.
        self._credits = (
            _AdaptiveCredits(ceiling=max_inflight, occupancy_low=occupancy_low, occupancy_high=occupancy_high)
            if adaptive_inflight
            else None
        )
        self._wake = asyncio.Event()
        self._completions: dict[int, tuple[asyncio.Future, float]] = {}
        self._reaper_task: asyncio.Task | None = None
        self._last_logged_target = 1
        self._batch_timeout = forward_timeout * num_stages + backward_timeout * max(1, num_stages - 1) + 30.0
        self._stall_timeout = max(forward_timeout, backward_timeout) + 30.0
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(prefetch_batches)))
        self._stop = threading.Event()
        self._seq = itertools.count(1)
        self._last_seq = 0
        self._orphan_candidates: set[int] = set()
        self._sweep_countdown = _ORPHAN_SWEEP_TICKS
        self._counters: dict[str, int] = {}
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._threads: list[threading.Thread] = []
        self._p2p: P2P | None = None
        self._own_peer_id: str | None = None
        self._this_maddrs: list[str] = []
        self._data_start_timeout = float(data_start_timeout)
        if self._data_start_timeout <= _DOWNLOAD_RETRY_BUDGET_S:
            logger.warning(
                f"[W2WInject] data_start_timeout {self._data_start_timeout:g}s is within the "
                f"{_DOWNLOAD_RETRY_BUDGET_S:g}s download retry budget: startup may abort while "
                "the first fetch is still retrying"
            )
        self._first_batch_ready = threading.Event()
        self._terminal_failure = threading.Event()
        self._terminal_error: Exception | None = None
        self._paused_for_ghost_phase1 = False

    def start(self) -> None:
        self._loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=self._loop.run_forever, name="w2w-inject-loop", daemon=True)
        loop_thread.start()
        self._threads.append(loop_thread)

        # A private replica: the process-cached one is bound to whichever loop first
        # created it, and awaiting a foreign-loop replica hangs until the RPC timeout.
        self._p2p = self._run(self._dht.replicate_p2p(fresh=True))
        self._own_peer_id = self._dht.peer_id.to_base58()
        self._this_maddrs = [str(addr) for addr in self._run(self._p2p.get_visible_maddrs())]

        data_thread = threading.Thread(target=self._stream_batches, name="w2w-inject-data", daemon=True)
        data_thread.start()
        self._threads.append(data_thread)

        self._wait_for_first_batch()

        asyncio.run_coroutine_threadsafe(self._inject_forever(), self._loop)
        logger.info(
            f"[W2WInject] origin {self._expert_uid} ready: shard {self._data_idx}, "
            f"manifest {self._manifest_identity}, max_inflight {self._max_inflight}"
        )

    def _wait_for_first_batch(self) -> None:
        if not self._first_batch_ready.wait(timeout=self._data_start_timeout):
            error = TimeoutError(f"Pithos produced no first batch within {self._data_start_timeout:g} seconds")
            self._record_data_failure(error)
            raise error
        with self._lock:
            terminal_error = self._terminal_error
        if terminal_error is not None:
            raise RuntimeError("Pithos data stream failed before its first batch") from terminal_error
        if self._stop.is_set():
            raise RuntimeError("Trainerless injector stopped before its first batch")

    def _record_data_failure(self, error: Exception) -> None:
        with self._lock:
            if self._terminal_error is not None:
                return
            self._terminal_error = error
        logger.error(f"[W2WInject] data stream failed: {type(error).__name__}: {error}")
        self._first_batch_ready.set()
        self._terminal_failure.set()

    def wait_for_terminal_failure(self) -> Exception | None:
        """Block until shutdown or a terminal reader failure, returning that failure."""
        self._terminal_failure.wait()
        with self._lock:
            return self._terminal_error

    def shutdown(self) -> None:
        self._stop.set()
        abort_downloads = getattr(self._batch_source, "abort", None)
        if abort_downloads is not None:
            abort_downloads()
        self._first_batch_ready.set()
        self._terminal_failure.set()
        if self._reaper_task is not None and self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._reaper_task.cancel)
        if self._p2p is not None and self._loop is not None and self._loop.is_running():
            with contextlib.suppress(Exception):
                asyncio.run_coroutine_threadsafe(self._p2p.shutdown(), self._loop).result(timeout=5.0)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        for thread in self._threads:
            thread.join(timeout=5.0)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def _ensure_reaper(self) -> None:
        if self._reaper_task is None or self._reaper_task.done():
            self._reaper_task = asyncio.get_running_loop().create_task(self._reap_forever())

    async def _reap_forever(self) -> None:
        """Resolve waiting injections from periodic ledger snapshots.

        One observer task serves every waiter, so completion observation uses no
        executor threads and a finished batch frees its credit on the next tick.
        """
        loop = asyncio.get_running_loop()
        while True:
            # A tick failure must not kill the task: reaping is the only path that
            # frees credits, and the respawn in _ensure_reaper needs an injection,
            # which a full window blocks.
            try:
                if self._completions:
                    snap = await loop.run_in_executor(None, self._ledger.snapshot)
                    now = time.monotonic()
                    for seq, (waiter, deadline) in list(self._completions.items()):
                        if waiter.done():
                            self._completions.pop(seq, None)
                            continue
                        base = (self._expert_uid, seq)
                        if snap.get((*base, "dropped")):
                            waiter.set_result(("dropped", None, str(snap.get((*base, "reason"), ""))))
                        elif snap.get((*base, "done")):
                            loss = snap.get((*base, "loss"))
                            waiter.set_result(("done", float(loss) if loss is not None else None, ""))
                        else:
                            progress = snap.get((*base, "progress"))
                            if progress is not None and now - progress > self._stall_timeout:
                                waiter.set_result(("dropped", None, f"no w2w progress for {self._stall_timeout:.1f}s"))
                            elif now > deadline:
                                waiter.set_result(("timeout", None, f"no completion after {self._batch_timeout:.1f}s"))
                        if waiter.done():
                            self._completions.pop(seq, None)
                self._sweep_countdown -= 1
                if self._sweep_countdown <= 0:
                    self._sweep_countdown = _ORPHAN_SWEEP_TICKS
                    await self._sweep_orphans(loop)
            except Exception as e:
                logger.warning(f"[W2WInject] reaper tick failed (retrying): {type(e).__name__}: {e}")
            await asyncio.sleep(_REAP_TICK_S)

    async def _sweep_orphans(self, loop: asyncio.AbstractEventLoop) -> None:
        """Drop ledger keys resurrected by a late report racing drop_slot.

        report_event's membership check and field writes are separate manager
        round-trips, so a late report can re-create keys for an already-dropped
        batch. A seq is condemned only when it has no waiter, its progress stamp
        is older than the stall timeout, and both held on the previous sweep too.
        """
        snap = await loop.run_in_executor(None, self._ledger.snapshot)
        active = set(self._completions)
        now = time.monotonic()
        stale = set()
        for seq in {key[1] for key in snap}:
            if seq in active or seq > self._last_seq:
                continue
            progress = snap.get((self._expert_uid, seq, "progress"))
            if progress is None or now - progress > self._stall_timeout:
                stale.add(seq)
        confirmed = stale & self._orphan_candidates
        self._orphan_candidates = stale - confirmed
        if not confirmed:
            return

        def _drop_confirmed() -> None:
            for seq in confirmed:
                self._ledger.drop_slot(seq)

        await loop.run_in_executor(None, _drop_confirmed)
        self._bump("orphan_slots_swept", len(confirmed))
        logger.warning(f"[W2WInject] swept {len(confirmed)} orphaned ledger slot(s): seqs {sorted(confirmed)[:8]}")

    def _run(self, coro, timeout: float = 60.0):
        # Bound startup calls: an unresponsive p2p daemon surfaces as a TimeoutError
        # instead of a wedged server thread.
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=timeout)

    def _stream_batches(self) -> None:
        try:
            for batch in self._batch_source():
                while not self._stop.is_set():
                    try:
                        self._queue.put(batch, timeout=1.0)
                        self._first_batch_ready.set()
                        break
                    except queue.Full:
                        continue
                if self._stop.is_set():
                    return
            if not self._stop.is_set():
                raise RuntimeError("Pithos data stream ended")
        except Exception as e:
            # An error surfacing after shutdown (e.g. a read whose retries were
            # aborted) is not a data failure, just the stream unwinding.
            if not self._stop.is_set():
                self._record_data_failure(e)
        finally:
            while not self._stop.is_set():
                try:
                    self._queue.put(None, timeout=1.0)
                    break
                except queue.Full:
                    continue

    def _next_batch(self):
        while not self._stop.is_set():
            try:
                return self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
        return None

    def _window(self) -> int:
        if self._credits is None:
            return self._max_inflight
        target = min(self._credits.target, self._max_inflight)
        if target != self._last_logged_target:
            logger.info(f"[W2WInject] origin {self._expert_uid} adaptive window -> {target}")
            self._last_logged_target = target
        return target

    async def _wait_until_injection_allowed(self) -> bool:
        while not self._stop.is_set() and self._training_coordinator.ghost_phase == GhostPhase.PHASE1:
            if not self._paused_for_ghost_phase1:
                logger.info(f"[W2WInject] origin {self._expert_uid} paused in ghost phase 1")
                self._paused_for_ghost_phase1 = True
            await asyncio.sleep(_GHOST_PHASE_POLL_S)

        if self._paused_for_ghost_phase1 and not self._stop.is_set():
            logger.info(
                f"[W2WInject] origin {self._expert_uid} resumed after ghost phase 1 "
                f"({self._training_coordinator.ghost_phase.name.lower()})"
            )
            self._paused_for_ghost_phase1 = False
        return not self._stop.is_set()

    async def _inject_forever(self) -> None:
        pending: set[asyncio.Task] = set()
        loop = asyncio.get_running_loop()

        def _on_task_done(task: asyncio.Task) -> None:
            pending.discard(task)
            self._wake.set()

        while not self._stop.is_set():
            if not await self._wait_until_injection_allowed():
                break
            batch = await loop.run_in_executor(None, self._next_batch)
            if batch is None:
                break
            while len(pending) >= self._window():
                # Re-check after clearing: a task finishing between the check above and the
                # clear would otherwise erase its own wakeup and park the loop forever.
                self._wake.clear()
                if len(pending) < self._window():
                    break
                await self._wake.wait()
            if not await self._wait_until_injection_allowed():
                break
            task = asyncio.create_task(self._inject_one(batch))
            pending.add(task)
            task.add_done_callback(_on_task_done)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        logger.info(f"[W2WInject] origin {self._expert_uid} injection loop exited")

    async def _inject_one(self, batch: PithosBatch | torch.Tensor) -> None:
        if not await self._wait_until_injection_allowed():
            return
        seq = next(self._seq)
        self._last_seq = seq
        started = time.monotonic()
        try:
            if isinstance(batch, PithosBatch):
                input_ids = batch.input_ids
                data_manifest = batch.manifest_identity
                sample_start = batch.sample_start
                sample_stride = batch.sample_stride
            else:
                input_ids = batch
                data_manifest = None
                sample_start = None
                sample_stride = None
            batch_size, tokens = input_ids.shape[0], input_ids.shape[1] - 1
            hidden_ids = input_ids[:, :-1].cpu().detach()
            labels = input_ids[:, 1:].cpu().detach()
            loss_weight = torch.full((batch_size, tokens), 1.0 / (batch_size * tokens), dtype=torch.float32)
            metadata = BatchMetadata(
                trainer_uid=self._expert_uid,
                seq=seq,
                data_shard=self._data_idx,
                trainer_peer_id=self._own_peer_id,
                data_manifest=data_manifest,
                sample_start=sample_start,
                sample_stride=sample_stride,
                sample_rows=batch_size if sample_start is not None else None,
            )
            if not await self._wait_until_injection_allowed():
                return
            self._ledger.new_slot(seq)
            metadata_bytes = metadata.to_bytes()
            status, reason, accepted = await self._driver.accept_forward(self._expert_uid, metadata_bytes, self._p2p)
            if status != W2WPushStatus.ACCEPTED or accepted is None or not accepted.is_new:
                self._ledger.drop_slot(seq)
                self._bump("inject_rejected")
                if self._credits is not None:
                    self._credits.on_congestion()
                await asyncio.sleep(0.5)
                return
            # Self-injection bypasses ConnectionHandler, so replicate its cache put here: in cache
            # mode the backward carries grads only, and without the stage-0 input cached under the
            # same key the head's own backward would miss and drop every batch.
            if self._activation_cache is not None:
                cache_key = self._activation_cache.key(self._expert_uid, metadata_bytes)
                if cache_key is not None:
                    self._activation_cache.put(cache_key, [hidden_ids])
            passthrough = (labels, loss_weight)
            passthrough_serialized = [
                serialize_torch_tensor(tensor, runtime_pb2.CompressionType.NONE) for tensor in passthrough
            ]
            # Registered before the forward: a push stalled in congestion must read as
            # in flight, or the orphan sweep would condemn the batch's live slot.
            self._ensure_reaper()
            waiter: asyncio.Future = asyncio.get_running_loop().create_future()
            self._completions[seq] = (waiter, time.monotonic() + self._batch_timeout)
            try:
                await self._driver.handle_forward(
                    uid=self._expert_uid,
                    accepted=accepted,
                    forward_inputs=(hidden_ids,),
                    passthrough=passthrough,
                    passthrough_serialized=passthrough_serialized,
                    expert=self._expert,
                    p2p=self._p2p,
                    this_peer_id=self._own_peer_id,
                    this_maddrs=self._this_maddrs,
                )
                outcome, loss, drop_reason = await waiter
            finally:
                self._completions.pop(seq, None)
            if outcome == "done":
                self._reporter.report_train_step(loss, seq)
                self._bump("completed")
                if self._credits is not None:
                    self._credits.on_done(time.monotonic() - started)
            else:
                self._bump("dropped")
                if self._credits is not None:
                    self._credits.on_congestion()
                logger.warning(f"[W2WInject] seq={seq} {outcome}: {drop_reason}")
                # Back off before the next batch: a stage with no reachable worker (e.g.
                # downstream still starting) would otherwise spin, burning the dataset.
                await asyncio.sleep(_DROP_BACKOFF_S)
        except Exception as e:
            self._bump("inject_errors")
            logger.error(f"[W2WInject] seq={seq} failed: {type(e).__name__}: {e}")
        finally:
            self._ledger.drop_slot(seq)
            if seq % _SUMMARY_EVERY == 0:
                occupancy = self._credits.last_occupancy if self._credits is not None else 0.0
                logger.info(
                    f"[W2WInject] window: {self._window()}, occupancy: {occupancy:.2f}, "
                    f"summary {self.stats()} ledger={self._ledger.stats()}"
                )

    def _bump(self, name: str, delta: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + delta
