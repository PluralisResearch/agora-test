# Copyright 2026 Pluralis Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Worker-side activation cache.

During the forward pass a worker caches the *input* activation of each microbatch, keyed by its
:class:`~agora_server.core.batch_metadata.BatchMetadata` identity. During the backward pass the
worker pops the cached input so the trainer only has to send the gradients (the worker recomputes
its forward from the cached input exactly as it does today from the trainer-resent input).

A worker runs several ``ConnectionHandler`` processes for one expert (``num_handlers`` is 4-8 in
practice), and a microbatch's forward and backward land on *different* handler processes of the same
worker. The cache must therefore be **shared across processes**: it is created once in the parent
:class:`~agora_server.core.server.server.Server` and inherited by the forked handlers via an
``mp.Manager``. Tensor bytes live in shared memory (``file_system`` sharing strategy, i.e. ``/dev/shm``),
so cross-process puts/pops pass handles, not data copies.

The cache is bounded (``max_entries`` + ``max_bytes`` + ``ttl_seconds``) with FIFO eviction so it
cannot grow without bound when a microbatch's backward never arrives (dropped batch, trainer crash).

Storage is pluggable via :class:`Store` so the bytes can later live on CPU (default), GPU, or SSD.
"""

import multiprocessing as mp
import time

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional, Tuple

import torch

from agora_server.core.batch_metadata import BatchMetadata
from agora_server.hivemind.utils.logging import get_logger


logger = get_logger(__name__)


class Store(ABC):
    """Pluggable storage backend for cached activation tensors.

    ``put`` returns an opaque handle that is stored in the cache index; ``get`` resolves a handle back
    to tensors; ``drop`` releases the backing storage.
    """

    @abstractmethod
    def put(self, tensors: Sequence[torch.Tensor]): ...

    @abstractmethod
    def get(self, handle) -> tuple[torch.Tensor, ...]: ...

    def drop(self, handle) -> None:
        """Release a handle's backing storage. Default: rely on refcounting/GC."""

    @staticmethod
    def nbytes(tensors: Sequence[torch.Tensor]) -> int:
        return sum(t.element_size() * t.nelement() for t in tensors)


class SharedMemoryStore(Store):
    """CPU backend: tensors are cloned into shared memory so cross-process lookups pass handles.

    The clone is deliberate: it isolates the cached copy from (a) aliasing of the caller's tensors and
    (b) any in-place mutation by the RPC serializer (``serialize_torch_tensor(..., allow_inplace=True)``
    mutates under non-NONE compression). The handle *is* the shared tensor tuple; the cache index
    stores it directly, and torch's reducer pickles the shared-storage handle (filename), not the bytes.
    """

    def put(self, tensors: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        return tuple(self._to_shared(t) for t in tensors)

    def get(self, handle) -> tuple[torch.Tensor, ...]:
        return tuple(handle)

    @staticmethod
    def _to_shared(t: torch.Tensor) -> torch.Tensor:
        t = t.detach()
        if t.is_cuda:  # defensive: handler tensors are CPU, but never share a CUDA tensor this way
            t = t.cpu()
        t = t.clone()  # independent copy; clone() output is never already-shared
        t.share_memory_()
        return t


class ActivationCache:
    """Bounded, cross-process activation cache shared by a worker's connection-handler processes.

    Construct **once in the parent process** (before the handlers fork) so all handlers inherit the
    same ``mp.Manager`` proxies. Keyed by ``(uid, trainer_uid, seq)``; value is the cached input
    tensor tuple. ``pop`` removes on read (evict-on-success-or-miss).
    """

    def __init__(
        self,
        manager=None,
        *,
        max_entries: int,
        max_bytes: int,
        ttl_seconds: float,
        store: Store | None = None,
        name: str = "",
    ):
        if max_entries <= 0:
            raise ValueError(f"max_entries must be positive, got {max_entries}")
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be positive, got {max_bytes}")
        # Create the cross-process manager here (in the parent, before the handler processes fork) when
        # the caller does not supply one. Callers that own a manager (e.g. tests) may pass it in.
        self._owns_manager = manager is None
        self._manager = manager if manager is not None else mp.Manager()
        self._store = store if store is not None else SharedMemoryStore()
        self._max_entries = int(max_entries)
        self._max_bytes = int(max_bytes)
        self._ttl_seconds = float(ttl_seconds)
        self._name = name

        # Lightweight, insertion-ordered metadata (FIFO order + TTL/byte bookkeeping) kept separate
        # from the heavy tensor handles so eviction sweeps never reconstruct tensors cross-process.
        self._meta = self._manager.dict()  # (uid,tid,seq) -> (nbytes: int, insert_monotonic: float)
        self._data = self._manager.dict()  # (uid,tid,seq) -> in_handle
        self._counters = self._manager.dict()  # name -> int (includes running "bytes")
        self._lock = self._manager.Lock()

    # -- public API ---------------------------------------------------------------------------

    def key(self, uid: str, metadata: bytes) -> tuple[str, str, int] | None:
        """Build the cache key from a microbatch's serialized BatchMetadata.

        Lives here (core) so the base ConnectionHandler (hivemind) can build keys without importing
        BatchMetadata, avoiding a hivemind->core layering inversion. Returns ``None`` when metadata is
        absent or unparseable (the caller then treats the RPC as the legacy non-cached path)."""
        if not metadata:
            return None
        md = BatchMetadata.from_bytes(metadata)
        if md is None:
            return None
        return (uid, md.trainer_uid, md.seq)

    def put(self, key: tuple[str, str, int], inputs: Sequence[torch.Tensor]) -> None:
        # Clone into shared memory outside the lock (the expensive part); index update is under lock.
        in_handle = self._store.put(inputs)
        nbytes = self._store.nbytes(inputs)
        now = time.monotonic()
        with self._lock:
            if key in self._meta:
                # Forward retry re-puts the same microbatch: drop the stale copy first (FIFO re-insert).
                self._evict_one_locked(key, reason="overwrites")
            self._meta[key] = (nbytes, now)
            self._data[key] = in_handle
            self._bump_locked("bytes", nbytes)
            self._bump_locked("puts", 1)
            self._evict_locked(now)

    def pop(self, key: tuple[str, str, int]) -> tuple[torch.Tensor, ...] | None:
        with self._lock:
            meta = self._meta.pop(key, None)
            if meta is None:
                self._bump_locked("misses", 1)
                return None
            nbytes, _ = meta
            handle = self._data.pop(key, None)
            self._bump_locked("bytes", -nbytes)
            if handle is None:
                # _meta/_data inconsistency should never happen; treat as a miss rather than crash.
                self._bump_locked("misses", 1)
                return None
            self._bump_locked("hits", 1)
        return self._store.get(handle)

    def record_drop(self) -> None:
        """Count a dropped microbatch (the backward could not be served). Diagnostics only."""
        with self._lock:
            self._bump_locked("drops", 1)

    def stats(self) -> dict:
        with self._lock:
            s = {k: self._counters.get(k, 0) for k in _COUNTER_NAMES}
            s["entries"] = len(self._meta)
            return s

    def clear(self) -> None:
        with self._lock:
            for handle in self._data.values():
                self._store.drop(handle)
            self._meta.clear()
            self._data.clear()
            self._bump_locked("bytes", -self._counters.get("bytes", 0))

    def close(self) -> None:
        """Shut down the owned manager process (no-op if the manager was supplied by the caller)."""
        if self._owns_manager:
            self._manager.shutdown()

    # -- internals (callers below hold self._lock) --------------------------------------------

    def _evict_locked(self, now: float) -> None:
        # TTL sweep: entries are insertion-ordered == time-ordered, so stop at the first live entry.
        if self._ttl_seconds > 0:
            for key in list(self._meta.keys()):
                _, inserted = self._meta[key]
                if now - inserted < self._ttl_seconds:
                    break
                self._evict_one_locked(key, reason="evictions_ttl")
        # Size / byte bound: evict oldest until within both limits.
        while self._meta and (len(self._meta) > self._max_entries or self._counters.get("bytes", 0) > self._max_bytes):
            oldest = list(self._meta.keys())[0]
            self._evict_one_locked(oldest, reason="evictions_fifo")

    def _evict_one_locked(self, key: tuple[str, str, int], reason: str) -> None:
        nbytes, _ = self._meta.pop(key)
        handle = self._data.pop(key, None)
        if handle is not None:
            self._store.drop(handle)
        self._bump_locked("bytes", -nbytes)
        self._bump_locked(reason, 1)

    def _bump_locked(self, name: str, delta: int) -> None:
        self._counters[name] = self._counters.get(name, 0) + delta


_COUNTER_NAMES = (
    "puts",
    "hits",
    "misses",
    "evictions_fifo",
    "evictions_ttl",
    "overwrites",
    "drops",
    "bytes",
)
