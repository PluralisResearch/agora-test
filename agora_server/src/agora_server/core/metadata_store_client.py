# Copyright 2026 Pluralis Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Thin HTTP client for the Metadata Store API.

Speaks the msgpack wire contract (see ``metadata_store/.../routes/kv.py``) and
hides it behind put / get / mput / mget. The store applies the ``run:{run_id}:``
prefix and runs the validation pipeline; this client only encodes/decodes, maps
status codes, and retries transport-level failures once (see ``_request``).

The HTTP transport is synchronous (httpx.Client).
Asynchrony is layered on top by client-owned threads: writes via the FIFO
writer (``put_async``), reads via a small thread pool (``submit_read``) - so
RedisDHT callers on event loops never block on the transport, while the plain
put/get/mput/mget methods stay blocking for callers that want
completion-before-return. A caller may inject an ``http_client`` (e.g.
Starlette's TestClient bound to the in-process app) for testing instead of
opening a real connection.

Async writes: ``put_async`` enqueues a write onto a per-client FIFO writer
thread. FIFO ordering is necessary since the MS server is last-write-wins,
not max-expiry wins.
"""

import gzip
import logging
import multiprocessing.util as mp_util
import os
import queue as queue_module
import threading
import time

from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, cast
from urllib.parse import quote

import httpx
import msgpack


logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)

_MSGPACK = "application/msgpack"

_DEFAULT_TIMEOUT = httpx.Timeout(connect=3.0, read=5.0, write=5.0, pool=3.0)

# Async FIFO writer max size
# The expired-write drop in _writer_loop keeps queue length
# self-limited at ~write_rate x record_TTL (tens of entries) during outages of
# any duration; but if limit is reached, it degrades to the synchronous behavior
# instead of silently dropping coordination records.
_WRITE_QUEUE_MAX = 1024
# For draining on shutdown
_CLOSE_DRAIN_TIMEOUT_S = 5.0
_CLOSE_SENTINEL = object()


class _PendingWrite:
    """One queued async write. Mutable on purpose: a newer write to the same
    (key, subkey) supersedes the payload in place while queued (conflation).

    Every Swarm coordination write is a last-write-wins snapshot from a single
    sequential owner, so sending a stale queued version ahead of a fresh one is pure waste.
    """

    __slots__ = ("key", "subkey", "envelope", "expiration_time", "on_done", "taken")

    def __init__(self, key, envelope, subkey, expiration_time, on_done):
        self.key = key
        self.subkey = subkey
        self.envelope = envelope
        self.expiration_time = expiration_time
        self.on_done = on_done
        self.taken = False  # writer has snapshotted it; no longer conflatable


def _safe_done(on_done: "Callable[[bool], None] | None", ok: bool) -> None:
    if on_done is None:
        return
    try:
        on_done(ok)
    except Exception:
        logger.warning("Metadata Store async write on_done callback failed", exc_info=True)


# Async read pool. Reads need no FIFO ordering — the
# pool exists so event-loop callers (matchmaking's 1 Hz poll, the progress
# fetcher) never pay HTTP latency inline on the loop. Deliberately separate
# from the FIFO writer: reads must not queue behind writes.
_READ_POOL_WORKERS = 4


# Stats reporting:
#
# Accumulate per-call latency across all MetadataStoreClient instances in the process and
# emit one `[Metadata Store Client Stats]` line per minute (mean/max latency,
# request count, failure count) rather than one line per request.
#
# Uses a deque instead of locks. Every call appends a (latency, ok) sample to
# a deque; `collections.deque.append`/`popleft` are atomic, so many
# concurrent writers + a single draining reader need no lock. The per-minute
# flush is still call-triggered (no background thread, so fork stays trivial),
# and "exactly one thread flushes this window" is enforced lock-free by a
# single-item `_flush_token` deque: whichever thread wins `popleft()` owns the
# flush, drains the samples, logs, then returns the token; everyone else gets
# IndexError and moves on.
_STATS_WINDOW_S = 60.0
# Max size of deque which accumulates the request stats within a _STATS_WINDOW_S
# (if reached will evict the oldest entries)
_MAX_SAMPLES = 2400
_samples: "deque[tuple[float, bool]]" = deque(maxlen=_MAX_SAMPLES)  # (latency_s, ok)
# One entry per transport-error retry (see _request). Same atomic-deque pattern
# as _samples: appenders are lock-free, the window flusher drains and counts.
_retries: "deque[None]" = deque(maxlen=_MAX_SAMPLES)
# One entry per 4xx validation rejection (contract violation; logged at ERROR)
# and per async write dropped because its record expired before it could be
# sent (see _writer_loop). Same atomic-deque pattern.
_validation_rejects: "deque[None]" = deque(maxlen=_MAX_SAMPLES)
_expired_drops: "deque[None]" = deque(maxlen=_MAX_SAMPLES)
_flush_token: "deque[object]" = deque([object()])  # 1 permit = the right to flush
_last_stats_log = time.monotonic()


def _record_call(latency_s: float, ok: bool) -> None:
    global _last_stats_log

    _samples.append((latency_s, ok))  # atomic; lock-free hot path

    if time.monotonic() - _last_stats_log < _STATS_WINDOW_S:
        return
    try:
        token = _flush_token.popleft()  # atomic claim; only one winner
    except IndexError:
        return  # another thread already owns this window's flush
    try:
        now = time.monotonic()
        if now - _last_stats_log < _STATS_WINDOW_S:
            return  # lost the race to a flush that just happened
        _last_stats_log = now

        buffer_at_cap = len(_samples) >= _MAX_SAMPLES

        requests = failures = 0
        latency_sum_s = latency_max_s = 0.0
        while True:
            try:
                latency, sample_ok = _samples.popleft()  # atomic
            except IndexError:
                break
            requests += 1
            latency_sum_s += latency
            if latency > latency_max_s:
                latency_max_s = latency
            if not sample_ok:
                failures += 1

        retries = _drain_count(_retries)
        rejects = _drain_count(_validation_rejects)
        dropped = _drain_count(_expired_drops)

        mean_ms = (latency_sum_s / requests * 1000.0) if requests else 0.0
        max_ms = latency_max_s * 1000.0
        # NOTE: new fields must be APPENDED — the monitor regexes
        # (swarm/monitor/monitor_class.py, prometheus/prom_roles) match this
        # line with an unanchored re.search over the leading fields.
        logger.info(
            f"[Metadata Store Client Stats] Last 60s - requests: {requests}, "
            f"mean_latency: {mean_ms:.1f}ms, max_latency: {max_ms:.1f}ms, failures: {failures}, "
            f"retries: {retries}, rejects: {rejects}, dropped_expired: {dropped}"
        )
        if buffer_at_cap:
            logger.warning(f"Metadata Store client stats buffer full ({_MAX_SAMPLES})")
    finally:
        _flush_token.append(token)  # always return the permit, even on log error


def _drain_count(d: "deque[None]") -> int:
    n = 0
    while True:
        try:
            d.popleft()  # atomic
        except IndexError:
            return n
        n += 1


def _reset_stats_after_fork() -> None:
    """Reset the accumulator in a forked child (e.g. averager mp.Process children,
    which inherit RedisDHT and this client). Without this the child would re-report
    the parent's samples and, if the parent held the flush token at fork, the child
    would inherit an empty `_flush_token` and never flush. Fresh state per child."""
    global _samples, _retries, _validation_rejects, _expired_drops, _flush_token, _last_stats_log
    _samples = deque(maxlen=_MAX_SAMPLES)
    _retries = deque(maxlen=_MAX_SAMPLES)
    _validation_rejects = deque(maxlen=_MAX_SAMPLES)
    _expired_drops = deque(maxlen=_MAX_SAMPLES)
    _flush_token = deque([object()])
    _last_stats_log = time.monotonic()


os.register_at_fork(after_in_child=_reset_stats_after_fork)


@contextmanager
def _instrument(op: str, key: str):
    """Time one Metadata Store request and feed the per-minute stats accumulator.

    A raised exception (httpx transport error or MetadataStoreError on an
    unexpected status) counts as a failure; a normal 404 cache-miss returns
    without raising, so it is NOT a failure.

    Severity split: environmental failures (transport errors, timeouts, 5xx) log at WARNING;
    a 4xx is a validation/contract rejection and logs at ERROR.
    """
    t0 = time.perf_counter()
    ok = True
    try:
        yield
    except Exception as exc:
        ok = False
        if isinstance(exc, MetadataStoreExpiredError):
            _expired_drops.append(None)  # atomic; counted into the stats line
            logger.warning(f"Metadata Store {op} {key!r} dropped: {exc}")
            raise
        status = getattr(exc, "status_code", None)
        if status is not None and 400 <= status < 500:
            _validation_rejects.append(None)  # atomic; counted into the stats line
            logger.error(f"Metadata Store {op} {key!r} rejected: {type(exc).__name__}: {exc}")
        else:
            logger.warning(f"Metadata Store {op} {key!r} failed: {type(exc).__name__}: {exc}")
        raise
    finally:
        _record_call(time.perf_counter() - t0, ok)


def _pack(obj) -> bytes:
    return cast(bytes, msgpack.dumps(obj, use_bin_type=True))


def _unpack(data: bytes) -> Any:
    return msgpack.loads(data, raw=False)


class MetadataStoreError(RuntimeError):
    """Raised when the Metadata Store returns an unexpected status.

    ``status_code`` carries the HTTP status so callers (and ``_instrument``'s
    severity split) can distinguish validation rejections (4xx) from
    environmental failures (5xx) without parsing the message.
    """

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class MetadataStoreExpiredError(MetadataStoreError):
    """A write was rejected because its record expired before the server saw
    it (400 "expiration too soon"). Semantically identical to the client-side
    expired-write drop — a race against time, not a contract violation — so
    _instrument logs it at WARNING and counts it as dropped_expired."""

    def __init__(self, message: str):
        super().__init__(message, status_code=400)


# Transport errors that can strike before (or without us knowing whether) the
# server processed the request — most commonly the keep-alive race: the server
# closes an idle pooled connection just as we reuse it and httpx raises
# RemoteProtocolError ("Server disconnected without sending a response"). The
# 2026-06-11 tc-shaped run hit exactly this on 20/24 volunteers, killing their
# progress fetcher/reporter coroutines. One immediate retry on a fresh
# connection is the standard remedy (RFC 9112 §9.6; Go's net/http does it
# transparently) and converts the race from routine to negligible.
#
# Retrying WRITES is safe too, audited 2026-06-12 (docs/dht-wip/m3-m4-log.md):
# every Swarm write is a last-write-wins upsert whose body (value + absolute
# expiration) is fully precomputed and whose (key, subkey) has a single,
# sequentially-writing owner — so re-applying a write that DID reach Redis is
# a no-op, and the inline (pre-return) retry cannot reorder a writer's own
# writes. HTTP status codes are deliberately NOT retried: a status means the
# server responded, so the ambiguity this retry exists for is absent (and the
# TTL-bounds check could legitimately 400 a near-ttl_min write retried later).
# Timeouts are also excluded — retrying would stack another full timeout onto
# hot paths (matchmaking, averaging), and they are not the stale-connection
# failure class.
_RETRYABLE_TRANSPORT_ERRORS = (
    httpx.RemoteProtocolError,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
)


class MetadataStoreClient:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        http_client: httpx.Client | None = None,
        token: str | None = None,
        token_provider: "Callable[[], str | None] | None" = None,
        compress_requests: bool = False,
        timeout: "float | httpx.Timeout" = _DEFAULT_TIMEOUT,
        write_queue_max: int = _WRITE_QUEUE_MAX,
    ):
        self._base_url = base_url
        self._token = token
        # Per-request bearer source: AccessTokens are short-lived, so
        # managed-mode clients pass a MetadataStoreTokenProvider here instead of
        # a static `token`. Evaluated in _request; takes precedence over `token`.
        self._token_provider = token_provider
        # Request-body compression: gzip bodies >= 1 KB; The server must accept Content-Encoding: gzip.
        # Response decompression needs no flag (httpx sends Accept-Encoding and inflates transparently)
        self._compress_requests = compress_requests
        self._timeout = timeout
        self._rebuild_lock = threading.Lock()
        # Async FIFO writer state (lazily created on first put_async; PID-scoped
        # so a forked child gets a fresh empty queue + thread instead of the
        # parent's — same pattern as the fork-aware httpx client in _http()).
        self._write_queue_max = write_queue_max
        self._write_queue: queue_module.Queue | None = None
        self._writer_thread: threading.Thread | None = None
        self._writer_pid: int | None = None
        # Conflation map: (key, subkey) -> queued-but-not-taken _PendingWrite.
        self._pending: dict = {}
        self._pending_lock = threading.Lock()
        # Async read pool (lazily created on first submit_read; PID-scoped like
        # the writer).
        self._read_pool: ThreadPoolExecutor | None = None
        self._read_pool_pid: int | None = None
        self._closing = False
        # Fork safety for the per-instance locks: a fork can land while another
        # thread holds _rebuild_lock/_pending_lock, and the child would then
        # deadlock on first use (the PID checks alone still have to ACQUIRE
        # those locks). Reinit them in mp-forked children — the same mechanism
        # mp.Queue uses for its internals. (Complements, not replaces, the
        # PID-scoped queue/thread/pool rebuilds.)
        mp_util.register_after_fork(self, MetadataStoreClient._after_fork_reinit)
        if http_client is not None:
            # Caller-managed client (e.g. an in-process test client). Not
            # fork-managed — the injecting fixture owns its lifecycle/scope.
            self._client = http_client
            self._owns_client = False
            self._pid: int | None = None
        else:
            if base_url is None:
                raise ValueError("MetadataStoreClient requires base_url or http_client")
            self._client = self._new_client()
            self._owns_client = True
            self._pid = os.getpid()

    @staticmethod
    def _after_fork_reinit(client: "MetadataStoreClient") -> None:
        client._rebuild_lock = threading.Lock()
        client._pending_lock = threading.Lock()
        client._pending = {}
        client._write_queue = None
        client._writer_thread = None
        client._writer_pid = None
        client._read_pool = None
        client._read_pool_pid = None

    def _new_client(self) -> httpx.Client:
        assert self._base_url is not None  # only built for the owned-client path
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        # keepalive_expiry must stay BELOW the server's keep-alive (see metadata_store/main.py)
        # (uvicorn timeout_keep_alive=75 in metadata_store/main.py) so the
        # client discards idle connections before the server closes them —
        # that ordering is what prevents the stale-connection race.
        limits = httpx.Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=65.0)
        return httpx.Client(base_url=self._base_url, headers=headers, timeout=self._timeout, limits=limits)

    def _http(self) -> httpx.Client:
        """Return the httpx client, rebuilt if we've crossed a fork boundary.

        An ``httpx.Client``'s connection pool / sockets are not safe to share
        across ``fork`` — a child that inherits a parent's client would corrupt
        the shared connections. RedisDHT is routinely carried into mp.Process
        children (e.g. averager subprocesses), so for an owned client we detect a
        PID change and build a fresh client (the inherited sockets are left for
        the OS to reap; we do NOT close them, which would touch the parent's fds).
        """
        if self._owns_client and os.getpid() != self._pid:
            # Double-checked under a lock so concurrent first-post-fork requests
            # rebuild the client exactly once (not once per racing thread).
            with self._rebuild_lock:
                if os.getpid() != self._pid:
                    self._client = self._new_client()
                    self._pid = os.getpid()
        return self._client

    def close(self) -> None:
        # Stop the FIFO writer with a bounded drain: pending writes get
        # _CLOSE_DRAIN_TIMEOUT_S to flush (covers tombstones / final declares at
        # normal latency) but a dead Metadata Store can never hang shutdown —
        # undrained entries are resolved False and would be TTL-dead anyway.
        with self._rebuild_lock:
            self._closing = True
            q, t = self._write_queue, self._writer_thread
        if q is not None and t is not None and self._writer_pid == os.getpid() and t.is_alive():
            try:
                q.put_nowait(_CLOSE_SENTINEL)
            except queue_module.Full:
                pass  # writer is wedged behind a full queue; don't block shutdown
            t.join(_CLOSE_DRAIN_TIMEOUT_S)
            undrained = 0
            while True:
                try:
                    item = q.get_nowait()
                except queue_module.Empty:
                    break
                if item is _CLOSE_SENTINEL:
                    continue
                undrained += 1
                _safe_done(item.on_done, False)
            # Sweep the conflation map: every write registers here BEFORE its
            # queue.put, so a put_async racing close() (enqueued after the
            # drain above, behind a stopped writer) still gets its future
            # resolved. A double resolution (writer completed it concurrently)
            # is benign — MPFuture.set_result on a done future is swallowed.
            with self._pending_lock:
                stragglers = [e for e in self._pending.values() if not e.taken]
                self._pending = {}
            for entry in stragglers:
                undrained += 1
                entry.taken = True
                _safe_done(entry.on_done, False)
            if undrained:
                logger.warning(f"Metadata Store client closed with {undrained} undrained async writes")
        if self._read_pool is not None and self._read_pool_pid == os.getpid():
            # No wait: in-flight/queued reads finish on their own (bounded by
            # the request timeouts) and resolve their futures; nothing new is
            # accepted after this.
            self._read_pool.shutdown(wait=False)
        if self._owns_client:
            self._client.close()

    # -- async FIFO writer ----------------------------------------------------

    def put_async(
        self,
        key: str,
        envelope: bytes,
        subkey=None,
        *,
        expiration_time: float,
        on_done: Callable[[bool], None] | None = None,
    ) -> None:
        """Enqueue a write for the background FIFO writer thread.

        Returns immediately; ``on_done(ok)`` is invoked exactly once from the
        writer thread when the write completes (True), fails (False, already
        logged by ``_instrument``), or is dropped because its record expired
        while queued (False). Blocks only if the queue is full.

        Writes (each including its inline transport retry) execute strictly
        in enqueue order (FIFO), preserving the single-writer ordering.

        Conflation (see _PendingWrite): if an earlier write to the same
        (key, subkey) is still queued (not yet taken by the writer), its
        payload is superseded in place — the stale version's ``on_done``
        resolves False, the queue keeps one entry per key, and per-key order
        is preserved (at most one queued version behind at most one in-flight
        older version).
        """
        self._ensure_writer()
        pkey = (str(key), subkey)
        superseded_done = None
        new_entry = None
        with self._pending_lock:
            entry = self._pending.get(pkey)
            if entry is not None and not entry.taken:
                superseded_done, entry.on_done = entry.on_done, on_done
                entry.envelope = envelope
                entry.expiration_time = expiration_time
            else:
                new_entry = _PendingWrite(key, envelope, subkey, expiration_time, on_done)
                self._pending[pkey] = new_entry
        if superseded_done is not None:
            _safe_done(superseded_done, False)  # superseded by a fresher snapshot
        if new_entry is not None:
            # Outside the lock: put() may block when the queue is full, and the
            # writer needs the lock to take entries.
            self._write_queue.put(new_entry)

    def _ensure_writer(self) -> None:
        if self._closing:
            raise RuntimeError("MetadataStoreClient is closed")
        if self._writer_thread is not None and self._writer_pid == os.getpid() and self._writer_thread.is_alive():
            return
        with self._rebuild_lock:
            if self._closing:
                raise RuntimeError("MetadataStoreClient is closed")
            if self._writer_thread is None or self._writer_pid != os.getpid() or not self._writer_thread.is_alive():
                # Fresh queue on (re)create: a forked child must not replay the
                # parent's pending writes.
                self._write_queue = queue_module.Queue(maxsize=self._write_queue_max)
                with self._pending_lock:
                    self._pending = {}
                self._writer_thread = threading.Thread(
                    target=self._writer_loop,
                    args=(self._write_queue,),
                    name="MetadataStoreWriter",
                    daemon=True,
                )
                self._writer_pid = os.getpid()
                self._writer_thread.start()

    def submit_read(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` on the client's read thread pool (contract doc, phase 2).

        Used by ``RedisDHT.get(return_future=True)`` so event-loop callers
        never pay HTTP latency inline. Reads carry no ordering requirement, so
        this is a plain pool, deliberately separate from the FIFO writer.
        """
        self._ensure_read_pool()
        self._read_pool.submit(fn)

    def _ensure_read_pool(self) -> None:
        if self._read_pool is not None and self._read_pool_pid == os.getpid():
            return
        with self._rebuild_lock:
            if self._closing:
                raise RuntimeError("MetadataStoreClient is closed")
            if self._read_pool is None or self._read_pool_pid != os.getpid():
                self._read_pool = ThreadPoolExecutor(
                    max_workers=_READ_POOL_WORKERS, thread_name_prefix="MetadataStoreReader"
                )
                self._read_pool_pid = os.getpid()

    def _writer_loop(self, q: "queue_module.Queue") -> None:
        while True:
            item = q.get()
            if item is _CLOSE_SENTINEL:
                return
            with self._pending_lock:
                # Snapshot under the lock: a conflating put_async may have
                # replaced the payload since this entry was queued.
                item.taken = True
                if self._pending.get((str(item.key), item.subkey)) is item:
                    del self._pending[(str(item.key), item.subkey)]
                key, envelope, subkey = item.key, item.envelope, item.subkey
                expiration_time, on_done = item.expiration_time, item.on_done
            ok = False
            try:
                if expiration_time <= time.time():
                    # Dead: the record expired while queued — only happens when
                    # the store is unreachable or too slow for the write rate.
                    # Drop it here; a nearly-dead record that slips through and
                    # dies in transit comes back as a 400 "expiration too soon",
                    # which is classified as the same weather (WARNING +
                    # dropped_expired), so nothing is lost by keeping this
                    # check simple.
                    _expired_drops.append(None)  # atomic; counted into the stats line
                    logger.warning(f"Metadata Store async write to {key!r} dropped: record expired before send")
                else:
                    self.put(key, envelope, subkey=subkey)  # logs its own failures
                    ok = True
            except Exception:
                ok = False  # already logged (and classified) by _instrument
            _safe_done(on_done, ok)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _key_path(key: str) -> str:
        # Fully percent-encode the caller key; the server route is {key:path}
        # and Starlette URL-decodes it, so the original key round-trips.
        return f"/v1/kv/{quote(str(key), safe='')}"

    def _request(self, method: str, path: str, body: dict) -> httpx.Response:
        """Send one request, retrying ONCE on a retryable transport error.

        The retry calls ``self._http()`` again: httpx discards the connection
        that errored, so the second attempt goes out on a fresh socket (and a
        post-fork client rebuild is picked up too). Must stay inline — the
        single-writer retry-safety argument (see _RETRYABLE_TRANSPORT_ERRORS)
        requires the duplicate to land before the caller's next write.
        """
        content = _pack(body)
        headers = {"content-type": _MSGPACK}
        if self._compress_requests and len(content) >= 1024:
            content = gzip.compress(content, compresslevel=1)
            headers["content-encoding"] = "gzip"
        if self._token_provider is not None:
            bearer = self._token_provider()  # cached; refreshes only near expiry
            if bearer:
                headers["authorization"] = f"Bearer {bearer}"
        try:
            return self._http().request(method, path, content=content, headers=headers)
        except _RETRYABLE_TRANSPORT_ERRORS as exc:
            logger.warning(
                f"Metadata Store {method} {path}: transport error ({type(exc).__name__}: {exc}); retrying once"
            )
            _retries.append(None)  # atomic; counted into the per-minute stats line
            return self._http().request(method, path, content=content, headers=headers)

    def _post(self, path: str, body: dict) -> httpx.Response:
        return self._request("POST", path, body)

    # -- operations ---------------------------------------------------------

    def put(self, key: str, envelope: bytes, subkey=None) -> bool:
        body: dict = {"envelope": envelope}
        if subkey is not None:
            body["subkey"] = subkey
        with _instrument("PUT", str(key)):
            resp = self._request("PUT", self._key_path(key), body)
            if resp.status_code == 204:
                return True
            if resp.status_code == 400 and "expiration too soon" in resp.text:
                raise MetadataStoreExpiredError(f"PUT {key!r} -> 400: {resp.text}")
            raise MetadataStoreError(f"PUT {key!r} -> {resp.status_code}: {resp.text}", status_code=resp.status_code)

    def mput(self, writes: Sequence[dict]) -> None:
        # Keys are coerced to str (the server requires str keys; the PoC relied
        # on redis-py coercing ints to strings).
        body_writes = [{**w, "key": str(w["key"])} for w in writes]
        with _instrument("mput", f"{len(body_writes)} writes"):
            resp = self._post("/v1/kv:mput", {"writes": body_writes})
            if resp.status_code == 400 and "expiration too soon" in resp.text:
                raise MetadataStoreExpiredError(f"mput -> 400: {resp.text}")
            if resp.status_code != 204:
                raise MetadataStoreError(f"mput -> {resp.status_code}: {resp.text}", status_code=resp.status_code)

    def get(self, key: str) -> dict | None:
        """Return the decoded read response, or None if the key is absent.

        Response shape: ``{"kind": "single", "envelope": bytes}`` or
        ``{"kind": "dict", "entries": [{"subkey", "envelope"}, ...]}``.
        """
        with _instrument("get", str(key)):
            resp = self._post("/v1/kv:get", {"key": str(key)})
            if resp.status_code == 404:
                return None
            if resp.status_code == 200:
                return _unpack(resp.content)
            raise MetadataStoreError(f"get {key!r} -> {resp.status_code}: {resp.text}", status_code=resp.status_code)

    def mget(self, keys: Sequence[str]) -> list[tuple[str, bytes | None]]:
        """Bulk Pattern-A read. Returns (key, envelope_or_None) in request order."""
        with _instrument("mget", f"{len(keys)} keys"):
            resp = self._post("/v1/kv:mget", {"reads": [{"key": str(k)} for k in keys]})
            if resp.status_code != 200:
                raise MetadataStoreError(f"mget -> {resp.status_code}: {resp.text}", status_code=resp.status_code)
            results = _unpack(resp.content)["results"]
            return [(r["key"], r["envelope"]) for r in results]
