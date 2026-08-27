"""Torch-returning readers: sample_id -> a [rows, seq+1] LongTensor of tokens.

This module is the ONLY part of pithos that imports torch, and pithos
deliberately neither depends on nor selects PyTorch: the torch used here is
the one already present in the consumer's environment (training images have
one). The base package imports cleanly without torch ever installed.

The mapping is a pure function of sample_id, so the same id yields the same
tokens on any number of workers and on a single-device reference, by
construction. seq and seed are read-time parameters. Unbounded sample_ids
wrap by modulo (multi-epoch).

StreamShard reads BOTH manifest formats: the approved `pithos_stream_v1`
contract (flat '<i4' objects, a power-of-two chunk size from 2**26 through
2**30 logical tokens + 1 overlap, seq a power of two from 512 through 128K)
and, explicitly for compatibility, the legacy
`colonnade_stream_v1` corpora the imported golden vectors pin. It is the
reference reader, kept behavior-identical to those vectors.

PithosBatchSource is the public training-facing API: it owns one Corpus
(from pithos.runtime — NumPy-only) and serves absolute
sample-range reads as CPU int64 tensors.

Ported from pretrain-data/reader.py (StreamShard); see PROVENANCE.md. The
golden vectors in tests/test_stream_reader.py pin the legacy contract.
"""

from __future__ import annotations

import hashlib

from .cache import DEFAULT_BUDGET_BYTES, ChunkCache
from .manifest import Manifest, StreamLayout
from .runtime import Corpus


try:
    import torch
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "pithos.torch needs PyTorch, which pithos deliberately does "
        "not depend on or select — install torch in your own environment first."
    ) from e


class StreamShard:
    """Reader for streamable chunk corpora: a hash-ordered token stream cut
    into contiguous little-endian int32 chunk objects, each chunk_tokens
    tokens + a 1-token overlap tail, plus a manifest. The `sample_id ->
    tokens` mapping is a PURE function of sample_id, so every consumer of
    (manifest, seq, seed) sees byte-identical tokens for the same id, on any
    box.

    Two deliberate design points (pretrain-data wiki Data page, D1/D5):
    (1) Layout is seq-agnostic: sample j of chunk k is the contiguous slice
        `chunk[j*seq : j*seq+seq+1]` (the overlap token is what lets the last
        sample of each chunk close within one object), so `chunk_tokens % seq
        == 0` is the only constraint and a future seq re-reads the same bytes.
    (2) Shuffle is the D1 block-local scheme: chunk order is the fixed at-rest
        (hash) order, but samples WITHIN a chunk are read through a permutation
        seeded by (data_seed, epoch, chunk). The at-rest order is already a
        uniform global shuffle (the hash), so a chunk is a uniform sample of
        the whole corpus; the within-chunk permutation breaks up the adjacent
        same-document runs a long doc would otherwise produce. Unbounded
        sample_ids wrap by modulo AND advance the epoch, so epoch 2 gets a
        fresh within-chunk order.

    Chunks are fetched lazily, verified against the manifest sha256, and
    memory-mapped under a byte budget (`budget_bytes`, default 1 GiB), so a
    worker's local footprint is bounded, never the whole corpus.
    """

    def __init__(
        self,
        manifest_path: str,
        cache_dir: str,
        chunk_urls: dict[str, str],
        seq: int,
        seed: int,
        budget_bytes: int = DEFAULT_BUDGET_BYTES,
    ) -> None:
        man = Manifest.load(manifest_path)
        # layout (incl. the supported-seq gate) is computed BEFORE the cache
        # is constructed, so an unsupported seq never creates or downloads
        # anything.
        layout = man.layout(seq)
        self.seq = seq
        self.seed = seed
        self.eos = man.eos_id
        self.chunk_names = man.chunk_names
        self.spc = layout.samples_per_chunk
        self.spc_k = layout.per_chunk
        self.cum = layout.cumulative
        self.n = layout.total_samples
        self._layout: StreamLayout = layout
        self._chunks = ChunkCache(cache_dir, chunk_urls, man.chunks, budget_bytes)
        self._perm: dict[tuple[int, int], object] = {}  # (epoch,k) -> permutation

    def _perm_for(self, epoch: int, k: int) -> object:
        key = (epoch, k)
        p = self._perm.get(key)
        if p is None:
            import numpy as np

            # blake2b, not Python hash(): a STABLE seed across processes and
            # platforms (hash() is per-process salted). The BitGenerator is
            # pinned to PCG64 explicitly, not default_rng's implicit choice.
            digest = hashlib.blake2b(f"{self.seed}:{epoch}:{k}".encode(), digest_size=8).digest()
            s = int.from_bytes(digest, "big")
            p = np.random.Generator(np.random.PCG64(s)).permutation(self.spc_k[k])
            self._perm = {key: p}  # only the current chunk's perm is live
        return p

    def _sample(self, idx: int) -> torch.Tensor:
        epoch, pos = divmod(idx, self.n)
        # every non-final chunk holds exactly spc samples (asserted at init),
        # and the short final chunk holds <= spc, so pos // spc is the chunk
        # for all pos in [0, n): pos<(nc-1)*spc -> a full chunk; the remainder
        # -> chunk nc-1. No clamp needed.
        k = pos // self.spc
        local = pos - self.cum[k]
        j = int(self._perm_for(epoch, k)[local])  # type: ignore[index]
        mm = self._chunks.get(self.chunk_names[k])
        off = j * self.seq
        window = mm[off : off + self.seq + 1]
        if window.shape[0] != self.seq + 1:  # never silently short-sample
            raise ValueError(
                f"chunk {self.chunk_names[k]} sample {j}: got {window.shape[0]} tokens, want {self.seq + 1}"
            )
        return torch.from_numpy(window.astype("int64"))

    def ids(self, batch_id: int, rows: int) -> torch.Tensor:
        """Batch `batch_id` of the epoch stream: the `rows` consecutive
        samples `[batch_id * rows, batch_id * rows + rows)`, as a
        [rows, seq+1] LongTensor.

        This is the inherited LEGACY batch-index semantics — `batch_id`
        indexes fixed-size batches, it is NOT an absolute sample-range
        start, and it does not express lease allocation. The behavior is
        pinned by the golden vectors and does not change here.
        PithosBatchSource.read instead takes absolute sample-range starts
        matching lease allocation.

        Callers that measure data-fetch stalls should time this call — the
        lazy chunk download at a chunk boundary is the stall."""
        out = [self._sample(batch_id * rows + r) for r in range(rows)]
        return torch.stack(out)  # [rows, seq+1]


class PithosBatchSource:
    """The public training-facing batch source: one owned Corpus read as
    torch tensors.

    The source takes EXPLICIT OWNERSHIP of the Corpus it is constructed
    with: `close` (or leaving the context manager) closes that Corpus. Do
    not share one Corpus across sources, and do not use a Corpus directly
    after handing it to a source.

    Reads use ABSOLUTE sample-range semantics — `start` is an absolute
    sample index into the epoch stream (wrapping by modulo total_samples,
    advancing the epoch), matching lease allocation. This is deliberately
    NOT StreamShard's legacy `ids(batch_id, rows)` batch-index semantics.
    """

    def __init__(self, corpus: Corpus) -> None:
        self._corpus = corpus

    @property
    def identity(self) -> str:
        """Canonical corpus identity (the computed manifest sha256)."""
        return self._corpus.identity

    @property
    def total_samples(self) -> int:
        """Samples per epoch at this corpus's sequence length."""
        return self._corpus.total_samples

    @property
    def samples_per_chunk(self) -> int:
        """Samples in every non-final chunk at this sequence length."""
        return self._corpus.samples_per_chunk

    @property
    def chunk_count(self) -> int:
        """Number of chunk objects in the corpus."""
        return self._corpus.chunk_count

    @property
    def sequence_length(self) -> int:
        """Tokens per sample."""
        return self._corpus.sequence_length

    def read(self, start: int, rows: int) -> torch.Tensor:
        """Read `rows` consecutive samples from absolute sample index
        `start`, as a CPU torch.int64 tensor of shape
        [rows, sequence_length + 1] (each sample plus its overlap token);
        `rows == 0` returns an empty [0, sequence_length + 1] tensor.

        The owned Corpus is asked for exactly this absolute range — one
        `Corpus.read(start, rows)` call — and ONLY that selected int32
        NumPy batch is widened to int64 (torch.from_numpy on the returned
        batch, then .to); corpus objects and cached mmaps are never cast.

        Args:
            start: absolute sample index; a nonnegative int.
            rows: how many samples to read; a nonnegative int.

        Raises:
            CacheError: if the source is closed (the owned Corpus's close
                error) or the range arguments are invalid.
        """
        batch = self._corpus.read(start, rows)
        return torch.from_numpy(batch).to(dtype=torch.int64)

    def read_strided(self, start: int, stride: int, rows: int) -> torch.Tensor:
        """Read a strided sequence of absolute samples as a CPU tensor.

        Args:
            start: First absolute sample index; a nonnegative int.
            stride: Distance between sample indices; a positive int.
            rows: Number of samples to read; a nonnegative int.

        Returns:
            A CPU torch.int64 tensor shaped [rows, sequence_length + 1].

        Raises:
            CacheError: If the source is closed or an argument is invalid.
        """
        batch = self._corpus.read_strided(start, stride, rows)
        return torch.from_numpy(batch).to(dtype=torch.int64)

    def close(self) -> None:
        """Close the owned Corpus (shutting down prefetch and releasing its
        resident mmaps). Idempotent. Afterwards `read` fails with the
        Corpus's closed error."""
        self._corpus.close()

    def __enter__(self) -> PithosBatchSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
