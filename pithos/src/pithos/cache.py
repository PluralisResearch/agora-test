"""Chunk leasing cache for streamable chunk corpora.

Chunks are fetched lazily, verified, memory-mapped, and held under a
deterministic BYTE budget (default 1 GiB) that bounds the aggregate ON-DISK
leased objects for every process sharing a cache directory — so the shared
local footprint stays bounded across restarts, no matter how large the
corpus. NumPy-only; no torch anywhere in the read path below the tensor
hand-off.

Integrity and safety rules:

- Every chunk is validated against its manifest sha256 BEFORE it is
  published or mapped — a same-size corrupted object is never trained on.
- Chunk names are validated as plain relative basenames and every
  filesystem path is checked for containment inside cache_dir. Managed
  objects are inspected with lstat WITHOUT following symlinks: a
  manifest-named symlink (including an alias to another file inside the
  cache), directory, or special file is rejected, never verified, mapped,
  touched, counted, or removed through. The lock directory is validated as
  a real, contained directory, and lock files are opened dir-relative with
  O_NOFOLLOW. Temp files are claimed with O_CREAT|O_EXCL|O_NOFOLLOW and the
  download is written to that already-open descriptor, so a planted temp
  symlink cannot redirect writes.
- No startup sweep: valid leased objects survive restarts, are accounted
  against the budget, and are reused after re-verification.
- Publication is an atomic same-directory rename from a per-process temp
  file. Downloads are RESUMABLE: progress is journaled to a sidecar
  ({bytes, sha256-of-prefix, etag, last_modified}) written only after the
  bytes it vouches for are on disk, and a later attempt — any process,
  under the chunk's exclusive lock — adopts the partial only after
  re-hashing the vouched prefix, sends Range/If-Range (transport.py), and
  restarts from byte zero when the server ignores or invalidates the
  range. A complete-but-wrong download is wiped, never resumed. Partial
  bytes are never published.
- Concurrency uses interprocess advisory locks (fcntl.flock, POSIX) with a
  documented two-level order: the cache-wide BUDGET lock first, then a
  per-chunk victim lock. No code path ever acquires the budget lock while
  holding a chunk lock, so the scheme cannot deadlock. Distinct chunk
  downloads hold only their own per-chunk lock and stay concurrent;
  same-chunk downloads serialize. Deleting an already-published file
  another process has mapped is POSIX-safe (the inode lives until the last
  reference closes).
- The disk LRU is recency-ordered by file mtime (refreshed on every hit,
  so recency persists across restarts), ties broken by name for
  determinism. The whole scan/evict transaction runs under the budget
  lock, so aggregate accounting is linearizable across processes; a victim
  is subtracted only when it was actually removed or is already absent.
  Eviction only ever happens in leasing mode — a pre-placed local corpus
  is never touched. A single object larger than the budget is kept (the
  documented oversized exception) rather than thrashing; that is the only
  way managed bytes may exceed the budget without an error.

Derived from the StreamShard chunk handling in pretrain-data/reader.py with
the above hardening; see PROVENANCE.md.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import stat
import threading
import time

from collections import OrderedDict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import numpy as np

from . import transport
from .errors import CacheError, DownloadError
from .manifest import ChunkInfo, validate_chunk_name


TOKEN_BYTES = 4  # '<i4'
DEFAULT_BUDGET_BYTES = 1 << 30  # 1 GiB of on-disk leased chunk objects
# chunk names must start alphanumeric (validate_chunk_name), so this
# dotfile lock name can never collide with a per-chunk lock file
_BUDGET_LOCK = ".budget.lock"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha256_prefix(path: str, nbytes: int) -> str:
    """sha256 of the first `nbytes` of `path` (the empty hash for 0)."""
    h = hashlib.sha256()
    read = 0
    with open(path, "rb") as f:
        while read < nbytes:
            block = f.read(min(1 << 20, nbytes - read))
            if not block:
                break
            h.update(block)
            read += len(block)
    return h.hexdigest()


class ChunkCache:
    """Lazy verified fetch + mmap + byte-budgeted disk LRU over chunk objects.

    Two modes:
      - leasing (chunk_urls given): missing chunks are downloaded on demand;
        corrupt cached objects are removed (under the chunk's lock) and
        re-fetched; the set of leased objects on disk is bounded by
        `budget_bytes` via mtime-ordered LRU eviction.
      - pre-placed local (chunk_urls={}): chunks must already exist in
        cache_dir; a corrupt object raises CacheError and is left in place;
        nothing is ever deleted and no recency metadata is written.

    Args:
        cache_dir: directory holding chunk objects (created if missing).
        chunk_urls: chunk name -> URL for leasing mode; {} for pre-placed.
        chunks: validated ChunkInfo entries (from a Manifest) — the expected
            token count and sha256 every cached object is held to. Names
            must be unique.
        budget_bytes: aggregate byte budget for managed on-disk objects
            across processes sharing `cache_dir`, and a per-process budget
            for resident mmaps; a positive int, default 1 GiB.

    Raises:
        CacheError: on integrity failure, unsafe names/paths/objects, or
            bad config.
        KeyError: if a chunk has no URL and is not cached locally.
    """

    def __init__(
        self,
        cache_dir: str,
        chunk_urls: dict[str, str],
        chunks: Iterable[ChunkInfo],
        budget_bytes: int = DEFAULT_BUDGET_BYTES,
    ) -> None:
        self.cache_dir = str(cache_dir)
        self.chunk_urls = dict(chunk_urls)
        self._expected: dict[str, ChunkInfo] = {}
        for info in chunks:
            # defense in depth: never trust a caller-supplied name near the fs
            validate_chunk_name(info.name)
            if info.name in self._expected:
                raise CacheError(f"duplicate chunk entry {info.name!r}")
            if info.tokens < 1:
                raise CacheError(f"chunk {info.name!r}: invalid token count {info.tokens}")
            self._expected[info.name] = info
        if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int):
            raise CacheError(f"budget_bytes must be a positive int, got {budget_bytes!r}")
        if budget_bytes < 1:
            raise CacheError(f"budget_bytes {budget_bytes} < 1")
        self.budget_bytes = budget_bytes
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except OSError as e:
            raise CacheError(f"cannot create cache dir {cache_dir!r}: {e}") from e
        self._locks_dir = os.path.join(cache_dir, ".locks")
        # the lock directory must be a real directory INSIDE the cache —
        # a planted symlink must never redirect lock writes outside it
        if os.path.islink(self._locks_dir):
            raise CacheError(f"lock dir {self._locks_dir} is a symlink")
        try:
            os.makedirs(self._locks_dir, exist_ok=True)
        except OSError as e:
            raise CacheError(f"cannot create lock dir {self._locks_dir}: {e}") from e
        if not os.path.isdir(self._locks_dir):
            raise CacheError(f"lock dir {self._locks_dir} is not a directory")
        if os.path.realpath(self._locks_dir) != os.path.join(os.path.realpath(self.cache_dir), ".locks"):
            raise CacheError(f"lock dir {self._locks_dir} escapes the cache dir")
        self._leased = bool(self.chunk_urls)
        self._mu = threading.Lock()
        self._resident: OrderedDict[str, np.memmap] = OrderedDict()  # LRU: oldest first
        self._resident_bytes = 0
        self._closed = False
        # account for valid leased objects surviving a restart: the budget
        # binds from construction, not from the first download
        self._enforce_disk_budget(protected=None)

    # ------------------------------------------------------------ paths
    def _contained_path(self, name: str) -> str:
        """Join `name` onto the (realpath'd) cache_dir and prove containment.
        The final component is deliberately NOT resolved: symlinks are
        detected and rejected at the point of use via lstat, never followed."""
        base = os.path.realpath(self.cache_dir)
        path = os.path.join(base, name)
        if os.path.dirname(path) != base:
            raise CacheError(f"chunk path {name!r} escapes cache dir {base}")
        return path

    @staticmethod
    def _lstat_regular(path: str, name: str) -> os.stat_result | None:
        """lstat the final component WITHOUT following symlinks; None if
        absent. A managed object must be a real regular file — an alias
        (even to another file inside the cache), directory, or special file
        is rejected, never operated on through the name."""
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(st.st_mode):
            raise CacheError(
                f"chunk object {name!r} is not a regular file (symlink/directory/special) — refusing to use it"
            )
        return st

    # ------------------------------------------------------------ locking
    @contextmanager
    def _open_locked(self, lock_name: str) -> Iterator[None]:
        """Exclusive interprocess advisory lock on a file inside `.locks`.

        The lock file is opened relative to a freshly validated locks-dir
        descriptor, and both opens use O_NOFOLLOW: a symlink planted at
        `.locks` or at the lock file itself fails the open instead of
        redirecting it outside the cache."""
        try:
            dfd = os.open(self._locks_dir, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError as e:
            raise CacheError(f"lock dir {self._locks_dir} is not a real directory: {e}") from e
        try:
            # Darwin's openat returns a spurious ENOENT when another process
            # concurrently creates the same lock file; the pinned directory
            # is healthy, so retry a few times. Any other error (e.g. ELOOP
            # from a planted symlink) fails hard.
            attempts_left = 4
            while True:
                try:
                    fd = os.open(lock_name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o644, dir_fd=dfd)
                    break
                except FileNotFoundError:
                    attempts_left -= 1
                    if attempts_left == 0:
                        raise CacheError(f"cannot open lock file {lock_name!r} in {self._locks_dir}") from None
                    time.sleep(0.001)
                except OSError as e:
                    raise CacheError(f"lock file {lock_name!r} is unsafe: {e}") from e
        finally:
            os.close(dfd)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    @contextmanager
    def _locked(self, name: str) -> Iterator[None]:
        """Exclusive interprocess advisory lock for one chunk object."""
        with self._open_locked(f"{name}.lock"):
            yield

    @contextmanager
    def _budget_locked(self) -> Iterator[None]:
        """The cache-wide eviction-transaction lock.

        Lock order: budget lock FIRST, then per-chunk victim locks; no code
        path acquires the budget lock while holding a chunk lock."""
        with self._open_locked(_BUDGET_LOCK):
            yield

    # ------------------------------------------------------------ verify
    def _verify(self, path: str, info: ChunkInfo) -> bool:
        """Size (cheap, clear) then sha256 (authoritative)."""
        if os.path.getsize(path) != info.tokens * TOKEN_BYTES:
            return False
        return _sha256_file(path) == info.sha256

    @staticmethod
    def _mmap(path: str) -> np.memmap:
        # explicit little-endian: the at-rest format is '<i4' on every host
        return np.memmap(path, dtype=np.dtype("<i4"), mode="r")

    # ------------------------------------------------------------ disk LRU
    def _scan_managed(self) -> dict[str, dict[str, tuple[int, float]]]:
        """Managed on-disk objects, grouped per manifest chunk: the chunk's
        regular final file, its exact `.{name}.resume.json` sidecar, and its
        exact `.{name}.part.<decimal pid>` partials, each inspected with
        lstat (no symlink following), as chunk name -> member name ->
        (bytes, mtime). Unknown names, symlinks, directories, and special
        files are ignored, never counted or deleted as objects."""
        base = os.path.realpath(self.cache_dir)
        groups: dict[str, dict[str, tuple[int, float]]] = {}
        try:
            entries = os.listdir(base)
        except OSError:
            return groups
        for entry in entries:
            if entry in self._expected:
                name = entry
            elif entry.startswith("."):
                rest = entry[1:]
                if rest.endswith(".resume.json"):
                    name = rest[: -len(".resume.json")]
                else:
                    # split at the LAST ".part.": chunk names may contain dots
                    name, sep, pid = rest.rpartition(".part.")
                    if not sep or not pid.isdigit():
                        continue
                if name not in self._expected:
                    continue
            else:
                continue
            try:
                st = os.lstat(os.path.join(base, entry))
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode):
                groups.setdefault(name, {})[entry] = (st.st_size, st.st_mtime)
        return groups

    def _touch(self, name: str) -> None:
        """Persisted recency: refresh the object's mtime on use (leased
        mode only — pre-placed corpora are never mutated)."""
        with self._locked(name):
            path = self._contained_path(name)
            if self._lstat_regular(path, name) is None:
                return  # another process evicted it; nothing to refresh
            os.utime(path, follow_symlinks=False)

    def _enforce_disk_budget(self, protected: str | None) -> None:
        """Evict least-recently-used managed groups until managed on-disk
        bytes fit the budget. A group's size is the sum of its member sizes
        (final object, resume sidecar, partials); its recency is the newest
        member mtime; victims are chosen oldest-first, chunk name as
        tiebreak.

        The whole scan/evict transaction runs under the cache-wide budget
        lock (taken BEFORE any per-chunk victim lock — see lock order in
        the module docstring), so aggregate accounting is linearizable
        across processes. Members are unlinked only while still regular
        files (lstat, no symlink following); missing or non-regular members
        are skipped, and a real unlink failure raises CacheError rather
        than silently leaving the cache over budget. After each eviction
        the managed set is rescanned and the total recomputed, so raced or
        resized members are never subtracted from stale bookkeeping. Never
        runs in pre-placed mode. A single group larger than the budget is
        kept (documented exception)."""
        if not self._leased:
            return
        with self._budget_locked():
            managed = self._scan_managed()
            total = sum(size for members in managed.values() for size, _ in members.values())
            while total > self.budget_bytes:
                candidates = [
                    (max(mtime for _, mtime in members.values()), name)
                    for name, members in managed.items()
                    if name != protected
                ]
                if not candidates or (protected is None and len(managed) <= 1):
                    break  # only the protected/last (possibly oversized) group remains
                _, victim = min(candidates)
                with self._locked(victim):
                    for member in managed[victim]:
                        mpath = self._contained_path(member)
                        try:
                            st = os.lstat(mpath)
                        except OSError:
                            continue  # raced away already: same on-disk effect
                        if not stat.S_ISREG(st.st_mode):
                            continue  # symlink/directory/special: never touch
                        try:
                            os.remove(mpath)
                        except OSError as e:
                            raise CacheError(f"cannot evict chunk {victim!r} member {member!r}: {e}") from e
                managed = self._scan_managed()
                total = sum(size for members in managed.values() for size, _ in members.values())
                with self._mu:
                    if self._resident.pop(victim, None) is not None:
                        self._resident_bytes -= self._expected[victim].tokens * TOKEN_BYTES

    # ------------------------------------------------------------ fetch
    def _ensure(self, name: str) -> np.memmap:
        info = self._expected.get(name)
        if info is None:
            raise KeyError(f"unknown chunk {name!r} (not in manifest)")
        with self._locked(name):
            path = self._contained_path(name)
            if self._lstat_regular(path, name) is not None:
                if self._verify(path, info):
                    return self._mmap(path)
                if not self._leased:
                    raise CacheError(
                        f"pre-placed chunk {name} failed integrity verification "
                        "(size/sha256 vs manifest) — fix the local corpus; "
                        "pithos never deletes pre-placed files"
                    )
                os.remove(path)  # corrupt leased object, removed under the lock
            url = self.chunk_urls.get(name)
            if not url:
                raise KeyError(f"no URL for chunk {name} and not cached locally")
            self._fetch_locked(name, info, url)
            return self._mmap(path)

    # ------------------------------------------------------------ download
    def _claim_or_adopt(self, name: str, tmp: str, state_path: str, source_id: str) -> tuple[int, int, str | None]:
        """Claim the per-process temp path, adopting a predecessor's partial
        download when its sidecar vouches for it.

        Returns (fd, have, prefix_sha256): the temp file open for writing,
        the number of validated prefix bytes already in it, and that
        prefix's sha256. A candidate is adopted only when the sidecar's
        source_id equals the current manifest-object identity AND its first
        sidecar-`bytes` are re-hashed and match; anything unvouched,
        oversized, surplus, or journaled under a foreign identity is
        deleted. A planted symlink at ANY partial temp name fails the whole
        operation — writes never follow it.
        """
        expected = self._expected[name].tokens * TOKEN_BYTES
        base = os.path.realpath(self.cache_dir)
        prefix = f".{name}.part."
        candidates: list[str] = []
        try:
            entries = os.listdir(base)
        except OSError as e:
            raise CacheError(f"cannot scan cache dir {base}: {e}") from e
        for entry in entries:
            if not entry.startswith(prefix) or not entry[len(prefix) :].isdigit():
                continue
            p = os.path.join(base, entry)
            st = os.lstat(p)
            if stat.S_ISLNK(st.st_mode):
                raise CacheError(f"temp path {entry!r} is a symlink — refusing to write")
            if stat.S_ISREG(st.st_mode):
                candidates.append(p)
        state = transport.load_state(state_path)
        if state is not None and not 0 <= state["bytes"] <= expected:
            state = None  # claims more than a whole object: garbage
        if state is not None and state["source_id"] != source_id:
            state = None  # a different manifest object: never adopt its bytes
        adopted: str | None = None
        have = 0
        prefix_sha: str | None = None
        for cand in candidates:
            ok = False
            if adopted is None and state is not None and os.path.getsize(cand) >= state["bytes"]:
                sha = _sha256_prefix(cand, state["bytes"])
                if sha == state["sha256"]:
                    adopted, have, prefix_sha, ok = cand, state["bytes"], sha, True
            if not ok:
                transport.delete_quiet(cand)  # invalid or surplus partial
        if adopted is None:
            transport.delete_quiet(state_path)  # a sidecar with no partial is stale
            try:
                # claim the temp name atomically and write to the already-open
                # descriptor: a planted temp symlink can never redirect the write
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
            except FileExistsError:
                raise CacheError(
                    f"temp path for chunk {name!r} already exists (planted or stale) — refusing to write"
                ) from None
            except OSError as e:
                raise CacheError(f"cannot create temp file for chunk {name!r}: {e}") from e
            return fd, 0, None
        if adopted != tmp:
            os.replace(adopted, tmp)  # rename unlinks anything planted at tmp; never follows
        try:
            fd = os.open(tmp, os.O_RDWR | os.O_NOFOLLOW)  # resume re-hashes the prefix via pread
        except OSError as e:
            raise CacheError(f"adopted temp path for chunk {name!r} is unsafe: {e}") from e
        os.ftruncate(fd, have)  # drop bytes the sidecar never vouched for
        return fd, have, prefix_sha

    def _fetch_locked(self, name: str, info: ChunkInfo, url: str) -> None:
        """Download `name` under its lock, resuming any adoptable partial,
        and publish atomically. A resumable failure keeps the partial plus
        sidecar for the next attempt (any process); a complete-but-wrong or
        non-resumable download wipes both."""
        path = self._contained_path(name)
        tmp = self._contained_path(f".{name}.part.{os.getpid()}")
        state_path = self._contained_path(f".{name}.resume.json")
        source_id = transport.object_identity(info.tokens * TOKEN_BYTES, info.sha256)
        fd: int | None = None
        try:
            fd, have, prefix_sha = self._claim_or_adopt(name, tmp, state_path, source_id)
            transport.fetch(url, fd, have, info.tokens * TOKEN_BYTES, state_path, source_id, prefix_sha)
            os.close(fd)
            fd = None
            if not self._verify(tmp, info):
                raise CacheError(
                    f"downloaded chunk {name} failed integrity verification (size/sha256 vs manifest) — not published"
                )
            os.replace(tmp, path)
            transport.delete_quiet(state_path)
        except DownloadError as e:
            if fd is not None:
                os.close(fd)
            if e.resumable:
                raise  # partial + journal stay for the next attempt
            transport.delete_quiet(tmp)
            transport.delete_quiet(state_path)
            raise
        except BaseException:
            if fd is not None:
                os.close(fd)
            transport.delete_quiet(tmp)
            transport.delete_quiet(state_path)
            raise

    # ------------------------------------------------------------ public
    def get(self, name: str) -> np.memmap:
        """The mmap'd little-endian int32 chunk `name`, fetching and evicting
        as needed. Hits refresh both in-process and persisted (mtime)
        recency.

        Raises:
            CacheError: if the cache is closed.
        """
        validate_chunk_name(name)
        with self._mu:
            if self._closed:
                raise CacheError(f"cache is closed — cannot get {name!r}")
            hit = self._resident.get(name)
            if hit is not None:
                self._resident.move_to_end(name)
        if hit is not None:
            if self._leased:
                self._touch(name)
            return hit
        try:
            mm = self._ensure(name)
        except DownloadError as e:
            # the per-chunk lock unwound with the exception, so the budget
            # lock may be taken here without violating lock order; the
            # failed chunk's own partial group stays protected for resume
            if e.resumable and self._leased:
                self._enforce_disk_budget(protected=name)
            raise
        with self._mu:
            if self._closed:  # closed while we fetched: never publish post-close
                raise CacheError(f"cache is closed — cannot get {name!r}")
            existing = self._resident.get(name)
            if existing is not None:  # another thread beat us to it
                self._resident.move_to_end(name)
                return existing
            self._resident[name] = mm
            self._resident_bytes += self._expected[name].tokens * TOKEN_BYTES
            while self._resident_bytes > self.budget_bytes:
                victim = next((n for n in self._resident if n != name), None)
                if victim is None:
                    # a single object larger than the budget stays resident —
                    # bounded (only it remains), never an eviction livelock
                    break
                self._resident_bytes -= self._expected[victim].tokens * TOKEN_BYTES
                self._resident.pop(victim)  # drops the mmap only; the leased file
                # on disk is governed by the disk LRU, not by mmap residency
        if self._leased:
            self._touch(name)
            self._enforce_disk_budget(protected=name)
        return mm

    @property
    def resident(self) -> tuple[str, ...]:
        """Currently mapped chunk names, least- to most-recently used."""
        with self._mu:
            return tuple(self._resident)

    @property
    def resident_bytes(self) -> int:
        """Current resident mmap bytes in this process."""
        with self._mu:
            return self._resident_bytes

    def managed_bytes(self) -> int:
        """Current managed ON-DISK bytes: every regular member of every
        manifest chunk's group (final object, resume sidecar, partials)."""
        return sum(size for members in self._scan_managed().values() for size, _ in members.values())

    def close(self) -> None:
        """Drop every resident mmap reference and zero the resident byte
        accounting. Idempotent. Afterwards `get` raises CacheError,
        including for names that were resident before close. On-disk
        objects are untouched: close governs this process's mappings only."""
        with self._mu:
            self._closed = True
            self._resident.clear()
            self._resident_bytes = 0

    def __enter__(self) -> ChunkCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
