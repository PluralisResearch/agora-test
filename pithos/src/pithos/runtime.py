"""The streaming Corpus runtime: a validated manifest and its sample layout
bound to a byte-budgeted, integrity-verified chunk cache with optional
prefetch.

Everything that can fail validation fails in `__init__` — manifest load and
identity, sample layout, seed, and the locator set — before a ChunkCache
exists or a single byte is fetched. NumPy-only at the base; never torch (the
torch-returning reader lives in `pithos.torch`).
"""

from __future__ import annotations

import hashlib

from collections.abc import Mapping

import numpy as np

from .cache import ChunkCache
from .config import CacheConfig
from .errors import CacheError, ManifestError
from .manifest import Manifest
from .prefetch import Prefetcher


class Corpus:
    """A deterministic sample-id-addressed view over a chunk corpus.

    Args:
        manifest_path: local path to a manifest.json.
        chunk_locators: chunk name -> URL for leasing mode; an empty mapping
            for a pre-placed local corpus in `cache_config.cache_dir`.
        sequence_length: tokens per sample; must be supported by the
            manifest format.
        seed: shuffle seed; validated here, consumed by `read`.
        cache_config: cache directory, byte budget, and prefetch depth.
        expected_manifest_identity: optional canonical identity the loaded
            manifest must match.

    Raises:
        ManifestError: on manifest load/validation failure, a non-string or
            mismatched expected identity, or an unsupported sequence length.
        CacheError: on a bad seed, a locator set not exactly covering the
            manifest's chunks, a non-string or empty locator value, or cache
            construction failure.
    """

    def __init__(
        self,
        manifest_path: str,
        chunk_locators: Mapping[str, str],
        sequence_length: int,
        seed: int,
        cache_config: CacheConfig,
        expected_manifest_identity: str | None = None,
    ) -> None:
        manifest = Manifest.load(manifest_path)
        if expected_manifest_identity is not None and not isinstance(expected_manifest_identity, str):
            raise ManifestError(f"expected_manifest_identity must be a str, got {expected_manifest_identity!r}")
        if expected_manifest_identity is not None and manifest.identity != expected_manifest_identity:
            raise ManifestError(
                f"manifest identity {manifest.identity[:16]}… != expected "
                f"{expected_manifest_identity[:16]}… — wrong corpus"
            )
        layout = manifest.layout(sequence_length)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise CacheError(f"seed must be an int, got {seed!r}")
        locators = dict(chunk_locators)
        names = set(manifest.chunk_names)
        keys = set(locators)
        if locators and keys != names:
            raise CacheError(
                "chunk_locators must be empty (pre-placed local corpus) or cover "
                f"exactly the manifest's chunks (missing {sorted(names - keys)}, extra {sorted(keys - names)})"
            )
        for chunk_name, url in locators.items():
            if not isinstance(url, str) or not url:
                raise CacheError(f"locator for chunk {chunk_name!r} must be a nonempty str, got {url!r}")
        self._manifest = manifest
        self._layout = layout
        self._seed = seed
        self._perm_key: tuple[int, int] | None = None
        self._perm: np.ndarray | None = None
        self._prefetch_depth = cache_config.prefetch_depth
        self._cache = ChunkCache(cache_config.cache_dir, locators, manifest.chunks, cache_config.budget_bytes)
        self._prefetcher = (
            Prefetcher(self._cache.get, cache_config.prefetch_depth) if cache_config.prefetch_depth > 0 else None
        )
        self._closed = False

    @classmethod
    def from_uri(
        cls,
        corpus_uri: str,
        sequence_length: int,
        seed: int,
        cache_config: CacheConfig,
        expected_manifest_identity: str | None = None,
    ) -> Corpus:
        """Open an immutable published corpus from a local, HTTP(S), or S3 URI.

        ``corpus_uri`` may name the publication directory or its
        ``manifest.json``. Remote manifests are bounded, validated, and
        cached locally; chunk locators are derived from the verified
        manifest and fetched lazily by the normal cache path.

        Args:
            corpus_uri: Publication root URI, or its manifest URI.
            sequence_length: Supported read-time context length.
            seed: Deterministic within-object shuffle seed.
            cache_config: Local cache configuration.
            expected_manifest_identity: Optional immutable identity pin.

        Returns:
            An open corpus using the same absolute sample-range contract as
            the direct constructor.
        """
        from .publication import resolve_publication

        publication = resolve_publication(
            corpus_uri,
            cache_config.cache_dir,
            expected_manifest_identity=expected_manifest_identity,
        )
        return cls(
            publication.manifest_path,
            publication.chunk_locators,
            sequence_length,
            seed,
            cache_config,
            expected_manifest_identity=expected_manifest_identity,
        )

    @classmethod
    def from_name(
        cls,
        corpus_name: str,
        sequence_length: int,
        seed: int,
        cache_dir: str,
        registry_path: str | None = None,
    ) -> Corpus:
        """Open a registry-named published corpus (see `pithos.registry`).

        The entry supplies the URI (R2 or S3 endpoint parameters included),
        the immutable identity pin, and any reader cache overrides its
        geometry needs; `from_uri` semantics apply from there. Callers
        needing full cache control use `from_uri` directly.
        """
        from .registry import resolve_corpus

        entry = resolve_corpus(corpus_name, registry_path)
        if entry.uri is None:
            raise CacheError(
                f"corpus {corpus_name!r} has no uri in the registry; its location is "
                "supplied at run time — open it with Corpus.from_uri instead"
            )
        overrides: dict[str, int] = {}
        if entry.budget_bytes is not None:
            overrides["budget_bytes"] = entry.budget_bytes
        if entry.prefetch_depth is not None:
            overrides["prefetch_depth"] = entry.prefetch_depth
        return cls.from_uri(
            entry.uri,
            sequence_length,
            seed,
            CacheConfig(cache_dir=cache_dir, **overrides),
            expected_manifest_identity=entry.manifest_identity,
        )

    @property
    def identity(self) -> str:
        """Canonical corpus identity (the computed manifest sha256)."""
        return self._manifest.identity

    @property
    def total_samples(self) -> int:
        """Samples per epoch at this corpus's sequence length."""
        return self._layout.total_samples

    @property
    def samples_per_chunk(self) -> int:
        """Samples in every non-final chunk at this sequence length."""
        return self._layout.samples_per_chunk

    @property
    def chunk_count(self) -> int:
        """Number of chunk objects in the corpus."""
        return len(self._layout.per_chunk)

    @property
    def sequence_length(self) -> int:
        """Tokens per sample."""
        return self._layout.seq

    def _ensure_open(self) -> None:
        if self._closed:
            raise CacheError("corpus is closed")

    def _permutation(self, epoch: int, k: int) -> np.ndarray:
        """The within-chunk sample permutation for (epoch, chunk k), seeded
        exactly as StreamShard's. Only the current chunk's perm stays live."""
        key = (epoch, k)
        perm = self._perm
        if self._perm_key != key or perm is None:
            # blake2b, not Python hash(): a STABLE seed across processes and
            # platforms (hash() is per-process salted). The BitGenerator is
            # pinned to PCG64 explicitly, not default_rng's implicit choice.
            digest = hashlib.blake2b(f"{self._seed}:{epoch}:{k}".encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            perm = np.random.Generator(np.random.PCG64(value)).permutation(self._layout.per_chunk[k])
            self._perm, self._perm_key = perm, key
        return perm

    def _fetch(self, k: int) -> np.memmap:
        """The verified mmap of chunk `k`, consuming a prefetch warm when one
        is in flight (a failed warm re-raises here, on demand, exactly where
        a synchronous fetch would raise). Afterwards warm the next distinct
        objects in manifest order, wrapping — at most `_prefetch_depth` of
        them and never the current one. A dropped request (pipeline full)
        is fine: the next read fetches on demand."""
        names = self._manifest.chunk_names
        if self._prefetcher is None:
            return self._cache.get(names[k])
        mm = self._prefetcher.get(names[k])
        n = len(names)
        for ahead in range(1, min(self._prefetch_depth, n - 1) + 1):
            self._prefetcher.request(names[(k + ahead) % n])
        return mm

    def read(self, start: int, rows: int) -> np.ndarray:
        """Read `rows` consecutive samples from absolute sample index `start`.

        `start` is an ABSOLUTE sample index into the epoch stream (not a
        batch id): sample indices wrap by modulo `total_samples`, advancing
        the epoch, so a multi-epoch read sees a fresh within-chunk
        permutation per epoch. Each row is `sequence_length + 1` int32
        tokens (the sample plus the overlap token) — NumPy only, no torch
        and no int64 widening.

        Args:
            start: absolute sample index; a nonnegative int.
            rows: how many samples to read; a nonnegative int (0 returns an
                empty (0, sequence_length + 1) array).

        Returns:
            A (rows, sequence_length + 1) np.int32 array of tokens.

        Raises:
            CacheError: if the corpus is closed, `start` or `rows` is a
                bool, non-int, or negative, or a chunk yields a short
                window (never silently short-sample).
        """
        self._ensure_open()
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise CacheError(f"start must be a nonnegative int, got {start!r}")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
            raise CacheError(f"rows must be a nonnegative int, got {rows!r}")
        layout = self._layout
        seq = layout.seq
        out = np.empty((rows, seq + 1), dtype=np.int32)
        if rows == 0:
            return out
        names = self._manifest.chunk_names
        n = layout.total_samples
        written = 0
        while written < rows:
            epoch, pos = divmod(start + written, n)
            k = pos // layout.samples_per_chunk
            local = pos - layout.cumulative[k]
            run = min(rows - written, layout.per_chunk[k] - local)
            mm = self._fetch(k)  # one fetch per contiguous run within one chunk
            perm = self._permutation(epoch, k)
            for t in range(run):
                j = int(perm[local + t])
                window = mm[j * seq : j * seq + seq + 1]
                if window.shape[0] != seq + 1:
                    raise CacheError(f"chunk {names[k]} sample {j}: got {window.shape[0]} tokens, want {seq + 1}")
                out[written + t] = window
            written += run
        return out

    def read_strided(self, start: int, stride: int, rows: int) -> np.ndarray:
        """Read a strided sequence of absolute samples.

        Row ``r`` is absolute sample ``start + r * stride``. This is the
        deterministic data-parallel primitive: stream ``d`` of ``n`` reads
        ``read_strided(d, n, rows)`` and therefore cannot overlap another
        valid stream in the same run.

        Args:
            start: First absolute sample index; a nonnegative int.
            stride: Distance between sample indices; a positive int.
            rows: Number of samples to read; a nonnegative int.

        Returns:
            A (rows, sequence_length + 1) np.int32 array of tokens.

        Raises:
            CacheError: If the corpus is closed, an argument is invalid, or
                a chunk yields a short window.
        """
        self._ensure_open()
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise CacheError(f"start must be a nonnegative int, got {start!r}")
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            raise CacheError(f"stride must be a positive int, got {stride!r}")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
            raise CacheError(f"rows must be a nonnegative int, got {rows!r}")

        layout = self._layout
        seq = layout.seq
        out = np.empty((rows, seq + 1), dtype=np.int32)
        names = self._manifest.chunk_names
        active_key: tuple[int, int] | None = None
        mm: np.memmap | None = None
        perm: np.ndarray | None = None
        for row in range(rows):
            absolute_sample = start + row * stride
            epoch, pos = divmod(absolute_sample, layout.total_samples)
            k = pos // layout.samples_per_chunk
            local = pos - layout.cumulative[k]
            key = (epoch, k)
            if key != active_key:
                mm = self._fetch(k)
                perm = self._permutation(epoch, k)
                active_key = key
            assert mm is not None and perm is not None
            j = int(perm[local])
            window = mm[j * seq : j * seq + seq + 1]
            if window.shape[0] != seq + 1:
                raise CacheError(f"chunk {names[k]} sample {j}: got {window.shape[0]} tokens, want {seq + 1}")
            out[row] = window
        return out

    def close(self) -> None:
        """Shut down prefetch first, then release the cache's resident mmaps
        and drop the cached permutation. Idempotent. On-disk leased objects
        are untouched."""
        if self._closed:
            return
        self._closed = True
        if self._prefetcher is not None:
            self._prefetcher.close()
        self._cache.close()
        self._perm = None
        self._perm_key = None

    def __enter__(self) -> Corpus:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
