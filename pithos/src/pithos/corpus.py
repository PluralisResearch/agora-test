"""Deterministic corpus-build primitives for the streamable chunk corpus.

The corpus order is a RULE, not a file: key(doc) = blake2b-128(dump + "\\0" + id,
keyed with HASH_SEED); order = ascending key over a pinned source revision.
Because the keys are i.i.d. uniform, that order IS a uniform global shuffle,
defined before any byte is written — so a key-threshold prefix (a tranche) is
a uniform sample of the full corpus. Docs are tokenized by a caller-supplied
encoder, concatenated in key order into one contiguous little-endian int32
('<i4') stream, and cut into flat chunk objects of CHUNK_TOKENS logical tokens
with a one-token overlap tail, so any seq+1 sample lives inside exactly one
object for any supported seq dividing CHUNK_TOKENS.

Byte-determinism is enforced, not hoped: intermediates are key-sorted records
(output identical for any fleet size or work assignment) and the merge
asserts global key monotonicity. Everything here is pure Python/NumPy and
raises typed `CorpusError` (a ValueError) on violation — never assert or
SystemExit.

Approved Pithos default contract (`pithos_stream_v1`): flat '<i4' objects
of 2**26 logical tokens (256 MiB) + one overlap token. CHUNK_TOKENS below IS
that default. HASH_SEED / EOS_ID / VOCAB_CAP are the LEGACY reference pins of
the imported colonnade corpus (see recipes/ and PROVENANCE.md) — a new corpus
pins its own values via its recipe, deliberately, never as a drive-by.

Ported from pretrain-data/prep_corpus.py; see PROVENANCE.md.
"""

from __future__ import annotations

import hashlib
import heapq
import itertools
import struct

from collections.abc import Callable, Iterable, Iterator
from typing import Any, BinaryIO, Protocol

from .errors import CorpusError


HASH_SEED = b"colonnade-165"  # LEGACY reference-corpus pin (recipes/); new corpora pin their own
KEY_BYTES = 16
CHUNK_TOKENS = 1 << 26  # approved default: 2**26 logical tokens = 256 MiB '<i4' per chunk (+1 overlap)
EOS_ID = 128001  # <|end_of_text|> for the pinned legacy Llama-3.1 tokenizer
VOCAB_CAP = 128512
MAX_EOS_DROP_RATE = 0.005
_REC_HDR = struct.Struct("<I")  # token count; preceded by KEY_BYTES raw key


def key_of(dump: str, doc_id: str) -> bytes:
    """Corpus-ordering key of a document: blake2b-128 over dump + NUL + id."""
    return hashlib.blake2b(f"{dump}\x00{doc_id}".encode(), key=HASH_SEED, digest_size=KEY_BYTES).digest()


def t1_from_fraction(fraction: float) -> bytes:
    """Threshold key: keep doc iff key < t1 (big-endian compare = int compare).

    Raises:
        CorpusError: if fraction is not strictly between 0 and 1.
    """
    if not 0.0 < fraction < 1.0:
        raise CorpusError(f"tranche fraction {fraction} not in (0, 1)")
    return int(fraction * (1 << (8 * KEY_BYTES))).to_bytes(KEY_BYTES, "big")


# ---------------------------------------------------------------- records
class _ByteSink(Protocol):
    """Anything with a bytes write — BinaryIO or a hashing/counting shim."""

    def write(self, b: bytes, /) -> int: ...


def write_record(out: _ByteSink, key: bytes, n_tokens: int, raw: bytes) -> None:
    """Append one (key, int32 token bytes) record to a part stream.

    Raises:
        CorpusError: if the key is not KEY_BYTES long, n_tokens is negative,
            or raw is not exactly 4 * n_tokens bytes.
    """
    if len(key) != KEY_BYTES:
        raise CorpusError(f"record key is {len(key)} bytes, must be {KEY_BYTES}")
    if n_tokens < 0:
        raise CorpusError(f"record n_tokens {n_tokens} < 0")
    if len(raw) != 4 * n_tokens:
        raise CorpusError(f"record body is {len(raw)} bytes, n_tokens says {4 * n_tokens}")
    out.write(key)
    out.write(_REC_HDR.pack(n_tokens))
    out.write(raw)


def read_records(f: BinaryIO) -> Iterator[tuple[bytes, bytes]]:
    """Yield (key, raw int32-le token bytes) until EOF.

    Raises:
        CorpusError: on a truncated key, header, or body.
    """
    while True:
        key = f.read(KEY_BYTES)
        if not key:
            return
        if len(key) != KEY_BYTES:
            raise CorpusError("truncated record key")
        hdr = f.read(_REC_HDR.size)
        if len(hdr) != _REC_HDR.size:
            raise CorpusError("truncated record header")
        (n,) = _REC_HDR.unpack(hdr)
        raw = f.read(4 * n)
        if len(raw) != 4 * n:
            raise CorpusError("truncated record body")
        yield key, raw


# ---------------------------------------------------------------- tokenize
def tokenize_docs(
    docs: Iterable[tuple[str, str, str]],
    encode_batch: Callable[[list[str]], list[list[int]]],
    t1: bytes,
    eos: int = EOS_ID,
    vocab_cap: int = VOCAB_CAP,
    batch: int = 256,
) -> tuple[list[tuple[bytes, bytes]], dict[str, int]]:
    """(dump, id, text) docs -> key-sorted [(key, token_bytes)] for key < t1.

    Token stream per doc = encoder output (BOS from the tokenizer's
    post-processor) + appended EOS. Docs whose tokenization CONTAINS the EOS
    id are dropped and counted (ambiguous boundary; the caller enforces
    MAX_EOS_DROP_RATE loudly).

    Raises:
        CorpusError: if batch < 1, a token id is negative or >= vocab_cap, or
            two docs in one input share a (dump, id) key.
    """
    if batch < 1:
        raise CorpusError(f"tokenize batch size {batch} < 1")
    out: list[tuple[bytes, bytes]] = []
    stats = {"seen": 0, "selected": 0, "dropped_eos": 0, "tokens": 0}
    pend_keys: list[bytes] = []
    pend_txt: list[str] = []

    def flush() -> None:
        for k, ids in zip(pend_keys, encode_batch(pend_txt), strict=True):
            stats["seen"] += 1
            if eos in ids:
                stats["dropped_eos"] += 1
                continue
            if ids:
                if min(ids) < 0:
                    raise CorpusError(f"negative token id {min(ids)}")
                if max(ids) >= vocab_cap:
                    raise CorpusError(f"token id {max(ids)} >= vocab cap {vocab_cap}")
            full = [*ids, eos]
            stats["selected"] += 1
            stats["tokens"] += len(full)
            out.append((k, struct.pack(f"<{len(full)}i", *full)))
        pend_keys.clear()
        pend_txt.clear()

    for dump, doc_id, text in docs:
        k = key_of(dump, doc_id)
        if k >= t1:
            stats["seen"] += 1
            continue
        pend_keys.append(k)
        pend_txt.append(text)
        if len(pend_txt) >= batch:
            flush()
    if pend_txt:
        flush()
    out.sort(key=lambda r: r[0])
    for a, b in itertools.pairwise(out):
        if a[0] == b[0]:  # duplicate (dump,id) within a file — data surprise
            raise CorpusError(f"duplicate key {a[0].hex()} within one file")
    return out, stats


# ---------------------------------------------------------------- chunker
class ChunkWriter:
    """Cut the merged token-byte stream into CHUNK_TOKENS-token objects with
    a one-token overlap tail: chunk k holds stream[k*C : (k+1)*C + 1], so
    every seq+1 sample (any seq dividing C) lives inside one object. The
    final chunk is partial and has no overlap tail.

    Raises:
        CorpusError: if chunk_tokens < 1 or a write is not a whole number of
            int32 tokens.
    """

    def __init__(self, chunk_tokens: int, sink: Callable[[int, bytes], None]) -> None:
        if chunk_tokens < 1:
            raise CorpusError(f"chunk_tokens {chunk_tokens} < 1")
        self.c = chunk_tokens
        self.sink = sink  # (chunk_index, chunk_bytes) -> None
        self.buf = bytearray()
        self.idx = 0
        self.total_tokens = 0
        self.chunks: list[dict[str, Any]] = []

    def write(self, token_bytes: bytes) -> None:
        if len(token_bytes) % 4 != 0:
            raise CorpusError(f"chunk write of {len(token_bytes)} bytes is not whole int32 tokens")
        self.total_tokens += len(token_bytes) // 4
        self.buf += token_bytes
        limit = 4 * (self.c + 1)
        while len(self.buf) >= limit:
            self._emit(bytes(self.buf[:limit]))
            # overlap: the last token of this chunk is also the first of the next
            del self.buf[: limit - 4]

    def close(self) -> None:
        if self.buf:
            self._emit(bytes(self.buf))
            self.buf.clear()

    def _emit(self, blob: bytes) -> None:
        name = f"chunk-{self.idx:05d}.bin"
        self.chunks.append({"name": name, "sha256": hashlib.sha256(blob).hexdigest(), "tokens": len(blob) // 4})
        self.sink(self.idx, blob)
        self.idx += 1


def merge_parts(
    parts: list[Iterator[tuple[bytes, bytes]]],
    writer: ChunkWriter,
    max_tokens: int | None = None,
) -> tuple[int, bytes | None]:
    """K-way merge of key-sorted parts into the chunk writer. Asserts global
    key monotonicity (catches an unsorted or corrupted part loudly).
    With max_tokens, the stream is cut at the EXACT logical-token boundary:
    the final record is truncated mid-document if needed, so the output
    holds exactly max_tokens logical tokens — a deterministic key-order
    prefix, i.e. a uniform sample of the full stream. Returns
    (records, last_key).

    Raises:
        CorpusError: if a part is not key-sorted or a key appears twice.
    """
    last: bytes | None = None
    n = 0
    for key, raw in heapq.merge(*parts, key=lambda r: r[0]):
        if last is not None and key < last:
            raise CorpusError("merge order violation — part not key-sorted")
        if key == last:
            raise CorpusError(f"duplicate 128-bit key {key.hex()} — investigate")
        if max_tokens is not None:
            remaining = max_tokens - writer.total_tokens
            if remaining <= 0:
                break
            if len(raw) > 4 * remaining:
                writer.write(raw[: 4 * remaining])
                n += 1
                last = key
                break
        last = key
        writer.write(raw)
        n += 1
    writer.close()
    return n, last


def numpy_sorted_records(bufs: list[bytes], selected: int) -> Iterator[tuple[bytes, bytes]]:
    """Same total key order as the heapq merge, produced by a single C-speed
    sort instead of a Python heap over the part streams. Parts are already in
    RAM (`bufs`); parse record locations, lexsort the 128-bit keys (as
    big-endian hi/lo u64 — EXACT bytewise order, avoiding numpy's S16
    trailing-NUL folding), then yield (key, token_bytes) in order. Feeding
    this single pre-sorted stream through merge_parts keeps the identical
    monotonicity/dup asserts and ChunkWriter, so output bytes are identical
    to the heapq path by construction.

    Raises:
        CorpusError: if a part buffer is truncated, holds more records than
            `selected`, or the parsed record count differs from `selected`.
    """
    import numpy as np

    if selected < 0:
        raise CorpusError(f"selected {selected} < 0")
    keys = np.empty((selected, 16), dtype=np.uint8)
    pid = np.empty(selected, dtype=np.int32)
    tstart = np.empty(selected, dtype=np.int64)
    tlen = np.empty(selected, dtype=np.int32)
    r = 0
    for bi, buf in enumerate(bufs):
        off, blen = 0, len(buf)
        while off < blen:
            if r >= selected:
                raise CorpusError(f"part buffer {bi} holds more than {selected} records")
            if off + 16 + _REC_HDR.size > blen:
                raise CorpusError(f"truncated record header in part buffer {bi} at {off}")
            keys[r] = np.frombuffer(buf, np.uint8, 16, off)
            (n,) = _REC_HDR.unpack_from(buf, off + 16)
            tstart[r] = off + 16 + _REC_HDR.size
            tlen[r] = 4 * n
            if tstart[r] + 4 * n > blen:
                raise CorpusError(f"truncated record body in part buffer {bi} at {off}")
            pid[r] = bi
            off = int(tstart[r]) + 4 * n
            r += 1
    if r != selected:
        raise CorpusError(f"parsed {r} records != {selected} selected")
    hi = np.ascontiguousarray(keys[:, :8]).view(">u8").ravel()
    lo = np.ascontiguousarray(keys[:, 8:]).view(">u8").ravel()
    order = np.lexsort((lo, hi))  # primary hi, secondary lo → bytewise ascending
    for idx in order:
        i = int(idx)
        p, s, ln = int(pid[i]), int(tstart[i]), int(tlen[i])
        yield keys[i].tobytes(), bufs[p][s : s + ln]
