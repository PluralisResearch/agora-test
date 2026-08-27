"""Recipe-controlled corpus policy: selection, keyed order, validation split.

The corpus order is a RULE: doc_key = keyed BLAKE2b-128 over (crawl,
document_id) with the lock-pinned per-corpus seed; ascending key order IS
the uniform global shuffle. The validation reservation is a domain-separated
keyed hash of the document key, so the split is stable, seed-independent of
the ordering hash, and reproducible without any stored assignment.

FineWeb-Edu policy (pinned by recipes/fineweb_edu_pithos_v1.json): include
only data/CC-MAIN-*/*.parquet; exclude the standalone overlapping
sample/10BT, sample/100BT, sample/350BT subsets (the real upstream paths);
upstream filtering and per-crawl MinHash dedup preserved; cross-crawl
duplicates RETAINED (never global-dedup); the legacy dev repetition filter
is NEVER applied — and nothing here implements either, by design.
"""

from __future__ import annotations

import hashlib
import re

from functools import cache

from ..errors import BuildError


KEY_BYTES = 16


@cache
def _segment_pattern(pat: str) -> re.Pattern[str]:
    """Anchored segment-aware glob: `*` and `?` match WITHIN a path segment
    (never '/'); `**/` spans ZERO or more whole segments and a trailing or
    bare `**` spans any suffix. This is what makes the FineWeb rule
    `data/CC-MAIN-*/*.parquet` exact (deeper nested lookalikes do not match)
    while `sample/350BT/**` still excludes the whole top-level sample dir."""
    out: list[str] = []
    i = 0
    while i < len(pat):
        c = pat[i]
        if c == "*":
            if pat[i : i + 3] == "**/":
                out.append("(?:.*/)?")
                i += 3
            elif pat[i : i + 2] == "**":
                out.append(".*")
                i += 2
            else:
                out.append("[^/]*")
                i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def selected(name: str, include: tuple[str, ...], exclude: tuple[str, ...]) -> bool:
    """Source-item selection: included iff it matches at least one include
    rule and no exclude rule (anchored, segment-aware)."""
    return any(_segment_pattern(p).match(name) for p in include) and not any(
        _segment_pattern(p).match(name) for p in exclude
    )


def doc_key(seed: bytes, crawl: str, document_id: str) -> bytes:
    """The corpus-ordering key of a document: keyed BLAKE2b-128 over
    crawl + NUL + document_id. Same construction as corpus.key_of, with the
    per-corpus seed pinned by the lock instead of the legacy module pin."""
    if len(seed) != KEY_BYTES:
        raise BuildError(f"hash seed is {len(seed)} bytes, must be {KEY_BYTES}")
    return hashlib.blake2b(f"{crawl}\x00{document_id}".encode(), key=seed, digest_size=KEY_BYTES).digest()


def is_validation(domain: str, fraction: float, key: bytes) -> bool:
    """Domain-separated validation reservation: reserve the document iff
    BLAKE2b-64(domain + NUL + doc_key) < fraction * 2**64. Deterministic,
    domain-separated from the ordering hash, and exactly reproducible."""
    if not 0.0 < fraction < 1.0:
        raise BuildError(f"validation fraction {fraction} not in (0, 1)")
    if len(key) != KEY_BYTES:
        raise BuildError(f"document key is {len(key)} bytes, must be {KEY_BYTES}")
    h = hashlib.blake2b(domain.encode() + b"\x00" + key, digest_size=8).digest()
    return int.from_bytes(h, "big") < int(fraction * (1 << 64))


def crawl_of(item_name: str) -> str:
    """The crawl a source item belongs to: its CC-MAIN-* path component when
    present (FineWeb layout), else the item's first path component."""
    parts = item_name.split("/")
    for part in parts:
        if part.startswith("CC-MAIN-"):
            return part
    return parts[0] if len(parts) > 1 else item_name.rsplit(".", 1)[0]
