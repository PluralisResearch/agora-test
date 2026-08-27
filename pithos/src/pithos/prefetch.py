"""Bounded asynchronous prefetch over the chunk cache's fetch path.

A Prefetcher warms at most `depth` chunk objects ahead of the consumer on a
SINGLE worker thread, through the exact fetch callable the consumer uses
(ChunkCache.get) — so every prefetched byte passes the same manifest-sha256
verification, per-chunk interprocess locking, atomic publication, and disk
budget enforcement as an on-demand read. Prefetch therefore can neither
corrupt a concurrent download (same-chunk fetches serialize on the chunk's
lock; distinct chunks stay concurrent) nor defeat the byte-budgeted disk
LRU: enforcement runs inside the fetch itself, and the single-threaded pool
bounds the transient publish-before-enforcement window to one object.

Contract:

- BOUNDED: at most `depth` requests are in flight (queued, running, or
  ready-but-unclaimed). A request beyond capacity is dropped (returns
  False); the consumer's `get` then fetches synchronously, so nothing is
  ever lost — prefetch is strictly an optimization.
- FAILURES SURFACE ON DEMAND: a failed warm is captured in its Future,
  never raised on the worker thread. `get(name)` pops the future and
  re-raises exactly where a synchronous fetch would have raised; because
  the failed future is popped, a retry goes through the live fetch path
  and resumes the journaled partial.
- NO LEAKS: `close()` cancels queued futures, waits out the running fetch,
  joins the pool thread, and drops the prefetcher's own references to
  unclaimed results. A result the fetch path itself retains — a
  ChunkCache's resident mmaps — is released by the owning cache's
  close(), not by the prefetcher. close is idempotent. `get` after close
  still works (synchronous path); `request` after close raises
  RuntimeError.

NumPy-free and torch-free.
"""

from __future__ import annotations

import threading

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Generic, TypeVar

from .errors import CacheError


_T = TypeVar("_T")

DEFAULT_PREFETCH_DEPTH = 2


class Prefetcher(Generic[_T]):
    """A depth-bounded, single-worker prefetch pipeline over `fetch`.

    Args:
        fetch: the consumer's own fetch callable (e.g. ChunkCache.get):
            name -> verified object. All integrity, locking, and budget
            rules live behind it.
        depth: maximum requests in flight; a positive int, default 2.

    Raises:
        CacheError: if depth is not a positive int.
    """

    def __init__(self, fetch: Callable[[str], _T], depth: int = DEFAULT_PREFETCH_DEPTH) -> None:
        if isinstance(depth, bool) or not isinstance(depth, int):
            raise CacheError(f"prefetch depth must be a positive int, got {depth!r}")
        if depth < 1:
            raise CacheError(f"prefetch depth {depth} < 1")
        self.depth = depth
        self._fetch = fetch
        self._mu = threading.Lock()
        self._inflight: dict[str, Future[_T]] = {}  # insertion-ordered
        self._closed = False
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pithos-prefetch")

    def request(self, name: str) -> bool:
        """Queue a warm of `name`. Returns False when the pipeline is full
        (the request is dropped; a later `get` fetches synchronously).
        Re-requesting an in-flight name is a cheap no-op returning True.

        Raises:
            RuntimeError: if the prefetcher is closed.
        """
        with self._mu:
            if self._closed:
                raise RuntimeError("prefetch request after close")
            if name in self._inflight:
                return True
            if len(self._inflight) >= self.depth:
                return False
            self._inflight[name] = self._pool.submit(self._fetch, name)
            return True

    def get(self, name: str) -> _T:
        """The fetched object, consuming the warm if one exists. A failed
        warm re-raises here — on demand, exactly where a synchronous fetch
        would raise — and is popped, so a retry resumes through the live
        fetch path. With no warm in flight this IS the synchronous path."""
        with self._mu:
            fut = self._inflight.pop(name, None)
        if fut is None:
            return self._fetch(name)
        return fut.result()

    @property
    def in_flight(self) -> tuple[str, ...]:
        """Names currently warmed or warming, oldest first. Never more
        than `depth` of them."""
        with self._mu:
            return tuple(self._inflight)

    def close(self) -> None:
        """Cancel queued warms, wait out the running fetch, join the pool
        thread, and drop this prefetcher's references to unclaimed results.

        Only the prefetcher's own references are released: a result the
        fetch callable itself retained (a ChunkCache's resident mmaps) is
        closed by the owning cache's close(), not here. Idempotent."""
        with self._mu:
            if self._closed:
                return
            self._closed = True
            unclaimed = list(self._inflight.values())
            self._inflight.clear()
        self._pool.shutdown(wait=True, cancel_futures=True)
        for fut in unclaimed:
            if not fut.cancelled():
                fut.exception()  # retrieve, then drop: the last reference goes away here

    def __enter__(self) -> Prefetcher[_T]:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
