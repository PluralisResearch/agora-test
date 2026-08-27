"""Tokenizer policies and the exact document encoding.

The encoding contract is EXACTLY [BOS] + tokens + [EOS] per document under
the locked tokenizer policy; a document whose own tokenization contains the
EOS id is dropped (ambiguous boundary) and counted, bounded by the recipe's
max_eos_drop_rate. BOS/EOS ids come from the recipe pins — never from
tokenizer introspection — so the encoding is stable across tokenizer
library versions. An interior BOS produced from source text is retained as
content: only EOS is an ambiguous document terminator in the flat stream.

The lock pins full tokenizer evidence: immutable revision, asset digest
(sha256 of tokenizer.json at the pinned revision), implementation version,
and the exact BOS/EOS/add-special-tokens policy. tokenizer_for verifies the
constructed policy against that evidence — a mismatch fails the build.

`ByteTokenizer` is the deterministic test-only policy (kind "byte").
Encoding is streamed in bounded batches: encode_stream yields EncodedDocs
in source order and never holds more than `batch` documents in memory.
"""

from __future__ import annotations

import hashlib
import struct

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from ..errors import BuildError
from . import policy as pol
from .recipe import Recipe
from .transforms import TransformSpec, apply_transforms


class TokenizerPolicy(Protocol):
    """A locked tokenizer: encodes text to token ids WITHOUT any BOS/EOS
    framing (pithos adds exactly [BOS] + tokens + [EOS] itself)."""

    name: str
    revision: str
    impl_version: str

    def encode_batch(self, texts: list[str]) -> list[list[int]]: ...


class ByteTokenizer:
    """Deterministic byte-level tokenizer for tests (recipe kind "byte").
    ids 0-15 are reserved; bos=1, eos=2; byte b encodes to b + 16."""

    name = "pithos-byte-test"
    revision = "v1"
    impl_version = "1"
    BOS_ID = 1
    EOS_ID = 2
    VOCAB_CAP = 272

    @classmethod
    def asset_sha256(cls) -> str:
        return hashlib.sha256(f"{cls.name}|{cls.revision}".encode()).hexdigest()

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [[b + 16 for b in t.encode("utf-8")] for t in texts]


def hf_tokenizer_evidence(name: str, revision: str | None, *, client: Any = None) -> dict[str, str]:
    """Lock-time HF tokenizer evidence: resolve the immutable commit, hash
    tokenizer.json at that commit, and pin the tokenizers implementation
    version. Fails clearly when the hub is unreachable — nothing invented."""
    if client is None:
        from .sources import HfHubClient  # noqa: PLC0415 — avoid a cycle at import time

        client = HfHubClient()
    try:
        commit = str(client.model_info(name, revision=revision).sha)
    except Exception as e:
        raise BuildError(f"cannot resolve tokenizer {name!r} revision {revision!r}: {e}") from e
    try:
        path = client.hf_hub_download(name, "tokenizer.json", repo_type="model", revision=commit)
    except Exception as e:
        raise BuildError(f"cannot fetch tokenizer.json for {name!r} at {commit}: {e}") from e
    try:
        import tokenizers  # noqa: PLC0415 — optional build dependency
    except ImportError as e:
        raise BuildError("huggingface tokenizers need the 'build' extra (tokenizers)") from e
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return {"revision": commit, "asset_sha256": h.hexdigest(), "impl_version": tokenizers.__version__}


class HuggingFaceTokenizer:
    """Locked HuggingFace tokenizer policy. Encoding uses the raw model (no
    post-processor) so the [BOS] + tokens + [EOS] framing is exactly
    pithos's. The tokenizer.json asset is fetched at the locked revision,
    re-hashed and checked against the lock, and the tokenizer is built from
    THAT EXACT VERIFIED FILE (Tokenizer.from_file — never an independent
    from_pretrained fetch); the installed tokenizers implementation version
    must equal the locked one."""

    def __init__(
        self,
        name: str,
        revision: str,
        expected_asset_sha256: str,
        expected_impl_version: str,
        client: Any = None,
    ) -> None:
        self.name = name
        self.revision = revision
        try:
            import tokenizers  # noqa: PLC0415 — optional build dependency

            from tokenizers import Tokenizer  # noqa: PLC0415
        except ImportError as e:
            raise BuildError("huggingface tokenizers need the 'build' extra (tokenizers)") from e
        self.impl_version = tokenizers.__version__
        if self.impl_version != expected_impl_version:
            raise BuildError(
                f"tokenizers implementation {self.impl_version} != locked {expected_impl_version} — "
                "install the locked version (tokenization is version-pinned)"
            )
        if client is None:
            from .sources import HfHubClient  # noqa: PLC0415

            client = HfHubClient()
        try:
            path = client.hf_hub_download(name, "tokenizer.json", repo_type="model", revision=revision)
        except Exception as e:
            raise BuildError(f"cannot fetch tokenizer.json for {name!r} at {revision!r}: {e}") from e
        try:
            with open(path, "rb") as stream:
                asset = stream.read()
        except OSError as e:
            raise BuildError(f"cannot read tokenizer {name!r} at {revision!r}: {e}") from e
        if hashlib.sha256(asset).hexdigest() != expected_asset_sha256:
            raise BuildError(f"tokenizer {name!r} asset digest mismatch at {revision} — lock evidence violated")
        try:
            self._tok = Tokenizer.from_str(asset.decode("utf-8"))
        except Exception as e:
            raise BuildError(f"cannot load tokenizer {name!r} at {revision!r}: {e}") from e
        self._tok.no_padding()
        self._tok.no_truncation()
        self._tok.post_processor = None  # pithos owns BOS/EOS framing exactly

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [list(enc.ids) for enc in self._tok.encode_batch(texts, add_special_tokens=False)]


def tokenizer_for(recipe: Recipe, tokenizer_lock: dict[str, Any], *, client: Any = None) -> TokenizerPolicy:
    """Construct the locked tokenizer policy and verify it against the
    lock's evidence (revision, asset digest, implementation version, and the
    exact BOS/EOS policy)."""
    if tokenizer_lock.get("name") != recipe.tokenizer_name:
        raise BuildError(
            f"lock tokenizer {tokenizer_lock.get('name')!r} != recipe tokenizer {recipe.tokenizer_name!r}"
        )
    if tokenizer_lock.get("bos_id") != recipe.bos_id or tokenizer_lock.get("eos_id") != recipe.eos_id:
        raise BuildError("lock BOS/EOS policy != recipe pins")
    if tokenizer_lock.get("add_special_tokens") is not False:
        raise BuildError("lock must pin add_special_tokens=False — pithos owns framing")
    if recipe.tokenizer_kind == "byte":
        tok: TokenizerPolicy = ByteTokenizer()
        if (
            tokenizer_lock.get("revision") != ByteTokenizer.revision
            or tokenizer_lock.get("asset_sha256") != ByteTokenizer.asset_sha256()
            or tokenizer_lock.get("impl_version") != ByteTokenizer.impl_version
        ):
            raise BuildError("byte tokenizer lock evidence mismatch")
        return tok
    return HuggingFaceTokenizer(
        recipe.tokenizer_name,
        str(tokenizer_lock["revision"]),
        str(tokenizer_lock["asset_sha256"]),
        str(tokenizer_lock["impl_version"]),
        client=client,
    )


@dataclass(frozen=True)
class EncodedDoc:
    key: bytes
    token_bytes: bytes  # little-endian int32 [BOS] + tokens + [EOS]
    validation: bool


def encode_stream(
    docs: Iterable[tuple[str, str, str]],
    tokenizer: TokenizerPolicy,
    *,
    seed: bytes,
    bos_id: int,
    eos_id: int,
    vocab_cap: int,
    validation_domain: str,
    validation_fraction: float,
    transforms: tuple[TransformSpec, ...] = (),
    batch: int = 256,
    stats: dict[str, int],
) -> Iterator[EncodedDoc]:
    """Stream (crawl, document_id, text) docs to EncodedDocs in SOURCE order
    (key-sorting is the worker's job, in bounded blocks).

    Per document: key = keyed BLAKE2b-128(crawl, id); validation =
    domain-separated reservation; tokens = [BOS] + encode(transforms(text))
    + [EOS]. Documents whose tokenization contains the EOS id are dropped
    and counted because EOS is the stream's document terminator. An interior
    BOS is retained as source content. `stats` accumulates
    seen/selected/dropped_eos/validation/tokens. Raises BuildError on a bad
    batch size or out-of-range token id.
    """
    if batch < 1:
        raise BuildError(f"tokenize batch size {batch} < 1")
    pend: list[tuple[bytes, bool]] = []
    pend_txt: list[str] = []

    def flush() -> Iterator[EncodedDoc]:
        for (k, is_val), ids in zip(pend, tokenizer.encode_batch(pend_txt), strict=True):
            stats["seen"] += 1
            if eos_id in ids:
                stats["dropped_eos"] += 1
                continue
            if ids:
                if min(ids) < 0:
                    raise BuildError(f"negative token id {min(ids)}")
                if max(ids) >= vocab_cap:
                    raise BuildError(f"token id {max(ids)} >= vocab cap {vocab_cap}")
            full = [bos_id, *ids, eos_id]
            stats["selected"] += 1
            stats["validation"] += int(is_val)
            stats["tokens"] += len(full)
            yield EncodedDoc(k, struct.pack(f"<{len(full)}i", *full), is_val)
        pend.clear()
        pend_txt.clear()

    for crawl, doc_id, text in docs:
        k = pol.doc_key(seed, crawl, doc_id)
        pend.append((k, pol.is_validation(validation_domain, validation_fraction, k)))
        pend_txt.append(apply_transforms(transforms, text))
        if len(pend_txt) >= batch:
            yield from flush()
    if pend_txt:
        yield from flush()


def check_eos_drop_rate(stats: dict[str, int], max_rate: float) -> None:
    """Enforce the recipe's EOS-drop bound loudly."""
    seen = stats["seen"]
    if seen and stats["dropped_eos"] / seen > max_rate:
        raise BuildError(
            f"EOS drop rate {stats['dropped_eos'] / seen:.4f} exceeds max_eos_drop_rate {max_rate} "
            f"({stats['dropped_eos']}/{seen} documents) — investigate the source or tokenizer"
        )


def records_of(docs: Iterator[EncodedDoc] | list[EncodedDoc]) -> Iterator[tuple[bytes, bytes]]:
    """(key, token_bytes) record stream for corpus.write_record/merge_parts."""
    for d in docs:
        yield d.key, d.token_bytes
