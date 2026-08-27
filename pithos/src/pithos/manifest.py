"""Manifest handling for streamable chunk corpora.

Two format tags exist side by side:

- `PITHOS_STREAM_V1` ("pithos_stream_v1") — the APPROVED Pithos contract:
  flat little-endian int32 ('<i4') chunk objects of a power-of-two logical
  token count from 2**26 (256 MiB, the build default) through 2**30 (4 GiB)
  + one overlap token, per-chunk sha256, and a verified canonical identity
  hash. Sequence length is a read-time parameter from the supported set:
  powers of two from 512 through 128K (131072).
- `LEGACY_STREAM_V1` ("colonnade_stream_v1") — explicit compatibility for the
  imported reference corpora (2**30-token chunks, dtype tag "int32", any
  positive seq dividing chunk_tokens). The imported golden vectors pin this
  path; it is not the new end state.

A manifest is validated end to end BEFORE anything is created or downloaded:
format, dtype, chunk geometry, per-chunk safe basename / token count /
sha256, structural chunk sizes, and the canonical identity hash (computed
over the manifest content and checked against the self-reported field — never
accepted on trust).

Ported from the StreamShard init-time validation in pretrain-data/reader.py
and hardened per the milestone-1 review; see PROVENANCE.md.
"""

from __future__ import annotations

import hashlib
import json
import re

from dataclasses import dataclass
from typing import Any

from .errors import ManifestError


PITHOS_STREAM_V1 = "pithos_stream_v1"
LEGACY_STREAM_V1 = "colonnade_stream_v1"
KNOWN_FORMATS = (PITHOS_STREAM_V1, LEGACY_STREAM_V1)

# The approved pithos_stream_v1 default is 2**26 logical tokens (256 MiB
# '<i4') per chunk + one overlap token; published corpora may use any
# power-of-two geometry from that default up to 2**30 (4 GiB).
PITHOS_CHUNK_TOKENS = 1 << 26
PITHOS_MAX_CHUNK_TOKENS = 1 << 30

# Approved read-time sequence lengths for pithos_stream_v1: powers of two
# from 512 through 128K inclusive (the small end serves debug models).
MIN_SEQ = 512
MAX_SEQ = 131072
SUPPORTED_SEQ = frozenset(1 << k for k in range(9, 18))

_CHUNK_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def validate_chunk_name(name: Any) -> str:
    """A chunk name must be a single safe relative basename.

    Rejects absolute paths, '..' traversal, embedded separators, dotfiles
    (which are the cache's lock/temp namespace), and anything outside
    [A-Za-z0-9._-]. Callers MUST still enforce containment after joining.

    Raises:
        ManifestError: if the name is not a safe basename.
    """
    if not isinstance(name, str) or not _CHUNK_NAME_RE.fullmatch(name):
        raise ManifestError(f"unsafe chunk name {name!r}: not a plain relative basename")
    if ".." in name:
        raise ManifestError(f"unsafe chunk name {name!r}: contains '..'")
    return name


def manifest_sha256(manifest: dict[str, Any]) -> str:
    """Canonical corpus identity: sha256 of the sorted-keys JSON of the
    manifest content, EXCLUDING any self-reported "manifest_sha256" field.
    This is computed, never accepted from the field itself."""
    content = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class ChunkInfo:
    """One validated chunk entry: safe basename, int32 token count, sha256."""

    name: str
    tokens: int
    sha256: str


@dataclass(frozen=True)
class StreamLayout:
    """Per-seq sample arithmetic over a manifest's chunks.

    Attributes:
        seq: sequence length this layout was computed for.
        samples_per_chunk: samples in every non-final chunk (chunk_tokens // seq).
        per_chunk: sample count of each chunk; the final chunk may be short.
        cumulative: cumulative sample-start per chunk, len n_chunks + 1.
        total_samples: samples per epoch (sample_ids wrap by modulo past this).
    """

    seq: int
    samples_per_chunk: int
    per_chunk: tuple[int, ...]
    cumulative: tuple[int, ...]
    total_samples: int


class Manifest:
    """Fully validated view of a stream-corpus manifest.

    Raises:
        ManifestError: on any structural, identity, or chunk-entry violation.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise ManifestError("manifest is not a JSON object")
        fmt = data.get("format")
        if fmt not in KNOWN_FORMATS:
            raise ManifestError(f"bad manifest format {fmt!r}")
        self.format = str(fmt)
        dtype = data.get("dtype")
        # Both tags mean little-endian int32 AT REST; the legacy tag predates
        # the explicit '<i4' spelling. Readers always mmap '<i4'.
        want_dtype = "<i4" if self.format == PITHOS_STREAM_V1 else "int32"
        if dtype != want_dtype:
            raise ManifestError(f"{self.format} expects dtype {want_dtype!r}, got {dtype!r}")
        eos_id = data.get("eos_id")
        chunk_tokens = data.get("chunk_tokens")
        if isinstance(eos_id, bool) or not isinstance(eos_id, int) or eos_id < 0:
            raise ManifestError(f"invalid eos_id {eos_id!r}")
        if isinstance(chunk_tokens, bool) or not isinstance(chunk_tokens, int):
            raise ManifestError(f"invalid chunk_tokens {chunk_tokens!r}")
        self.eos_id = eos_id
        self.chunk_tokens = chunk_tokens
        if self.chunk_tokens < 1:
            raise ManifestError(f"chunk_tokens {self.chunk_tokens} < 1")
        if self.format == PITHOS_STREAM_V1 and (
            self.chunk_tokens & (self.chunk_tokens - 1) != 0
            or not PITHOS_CHUNK_TOKENS <= self.chunk_tokens <= PITHOS_MAX_CHUNK_TOKENS
        ):
            raise ManifestError(
                f"{PITHOS_STREAM_V1} requires a power-of-two chunk_tokens between 2**26 "
                f"({PITHOS_CHUNK_TOKENS}) and 2**30 ({PITHOS_MAX_CHUNK_TOKENS}), got {self.chunk_tokens}"
            )

        raw_chunks = data.get("chunks")
        if not isinstance(raw_chunks, list) or not raw_chunks:
            raise ManifestError("manifest has no chunks")
        chunks: list[ChunkInfo] = []
        seen: set[str] = set()
        for entry in raw_chunks:
            if not isinstance(entry, dict):
                raise ManifestError(f"chunk entry is not an object: {entry!r}")
            name = validate_chunk_name(entry.get("name"))
            if name in seen:
                raise ManifestError(f"duplicate chunk name {name!r}")
            seen.add(name)
            raw_tokens = entry.get("tokens")
            if isinstance(raw_tokens, bool) or not isinstance(raw_tokens, int) or raw_tokens < 1:
                raise ManifestError(f"chunk {name!r}: invalid token count {raw_tokens!r}")
            tokens = raw_tokens
            sha = entry.get("sha256")
            if not isinstance(sha, str) or not _SHA256_RE.fullmatch(sha):
                raise ManifestError(f"chunk {name!r}: missing or malformed sha256")
            chunks.append(ChunkInfo(name=name, tokens=tokens, sha256=sha))
        # Structural chunk geometry (writer guarantee): every non-final chunk
        # is exactly chunk_tokens + 1 tokens (logical tokens + overlap tail);
        # the final chunk is partial, 1..chunk_tokens + 1.
        for c in chunks[:-1]:
            if c.tokens != self.chunk_tokens + 1:
                raise ManifestError(
                    f"non-final chunk {c.name!r} holds {c.tokens} tokens, "
                    f"must be chunk_tokens+1 ({self.chunk_tokens + 1})"
                )
        if chunks[-1].tokens > self.chunk_tokens + 1:
            raise ManifestError(
                f"final chunk {chunks[-1].name!r} holds {chunks[-1].tokens} tokens, "
                f"beyond chunk_tokens+1 ({self.chunk_tokens + 1})"
            )
        self.chunks = tuple(chunks)
        self.chunk_names = [c.name for c in self.chunks]
        self.chunk_token_counts = {c.name: c.tokens for c in self.chunks}

        # Identity is canonical: computed over the content, and the
        # self-reported field must exist and match — never accepted on trust.
        self.identity = manifest_sha256(data)
        reported = data.get("manifest_sha256")
        if not isinstance(reported, str) or not _SHA256_RE.fullmatch(reported):
            raise ManifestError("manifest missing or malformed manifest_sha256")
        if reported != self.identity:
            raise ManifestError(
                f"manifest_sha256 {reported[:16]}… != computed identity "
                f"{self.identity[:16]}… — corrupted or substituted manifest"
            )
        self.data = data

    @classmethod
    def load(cls, path: str) -> Manifest:
        """Load and validate a manifest.json from a local path."""
        with open(path) as f:
            return cls(json.load(f))

    def layout(self, seq: int) -> StreamLayout:
        """Sample layout at sequence length `seq`.

        Per-chunk sample count = (object_tokens - 1) // seq: a non-final
        chunk (object = chunk_tokens+1 with the overlap) yields exactly
        chunk_tokens//seq; the final chunk (no overlap tail) yields its own
        floor.

        The supported seq set is format-dependent: pithos_stream_v1 allows
        exactly the powers of two from 512 through 128K; the legacy format
        keeps the imported contract (any positive divisor of chunk_tokens)
        so the pinned golden vectors remain valid.

        Raises:
            ManifestError: if seq is not a positive int, is outside the
                supported set (pithos format), does not divide chunk_tokens,
                or the corpus holds zero samples at this seq.
        """
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
            raise ManifestError(f"seq {seq!r} is not a positive int")
        if self.format == PITHOS_STREAM_V1 and seq not in SUPPORTED_SEQ:
            raise ManifestError(
                f"seq {seq} not supported: {PITHOS_STREAM_V1} allows exactly "
                f"the powers of two {MIN_SEQ}..{MAX_SEQ} "
                f"({sorted(SUPPORTED_SEQ)})"
            )
        if self.chunk_tokens % seq != 0:
            raise ManifestError(
                f"chunk_tokens {self.chunk_tokens} not divisible by seq {seq} — "
                "the stream layout guarantees whole samples per chunk only "
                "for seq | chunk_tokens"
            )
        spc = self.chunk_tokens // seq  # samples in every non-final chunk
        spc_k = tuple((c.tokens - 1) // seq for c in self.chunks)
        for k, s in enumerate(spc_k[:-1]):
            if s != spc:
                raise ManifestError(f"chunk {k} yields {s} samples != {spc} (malformed manifest / wrong seq)")
        # final chunk must be <= a full chunk (writer guarantee); a larger one
        # would make `pos // spc` index past the last chunk (IndexError)
        if spc_k and spc_k[-1] > spc:
            raise ManifestError(f"final chunk yields {spc_k[-1]} > {spc} samples (malformed manifest)")
        cum = [0]
        for s in spc_k:
            cum.append(cum[-1] + s)
        n = cum[-1]
        if n <= 0:
            raise ManifestError("stream corpus holds zero samples at this seq")
        return StreamLayout(seq=seq, samples_per_chunk=spc, per_chunk=spc_k, cumulative=tuple(cum), total_samples=n)
