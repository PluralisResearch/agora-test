"""Recipe loading, validation, and the immutable lock phase.

A recipe is the corpus identity: it pins the format, geometry, tokenizer
policy, selection rules, validation reservation, ordering, transforms, and
output definitions. Pins that genuinely require resolution (source revision
and per-object content digests, tokenizer asset digest, code revision and
source-tree digest, per-corpus hash seed) are explicit nulls in the recipe
and are resolved ONLY by the lock phase — never silently invented. The lock
is create-once: re-locking with identical content is a no-op, locking with
different content fails.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile

from dataclasses import dataclass
from typing import Any

from ..errors import BuildError
from ..manifest import LEGACY_STREAM_V1, PITHOS_CHUNK_TOKENS, PITHOS_STREAM_V1
from .storage import check_http_base_url, check_inventory_names, check_object_name
from .transforms import TransformSpec, resolve_transforms, transform_identity_digest


FORMATS = {PITHOS_STREAM_V1, LEGACY_STREAM_V1}
SOURCE_KINDS = {"huggingface", "local", "s3", "http", "legacy_binary"}
TOKENIZER_KINDS = {"huggingface", "byte"}  # "byte" is the deterministic test-only tokenizer
LOCK_VERSION = 2


def _canonical_sha256(d: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class OutputSpec:
    """One named build output: a train or validation stream with its own
    format/geometry and an optional exact token budget (a deterministic
    key-order prefix of the global stream, cut at the exact token boundary)."""

    name: str
    role: str  # "train" | "validation"
    format: str
    chunk_tokens: int
    token_budget: int | None
    optional: bool
    description: str


@dataclass(frozen=True)
class Recipe:
    name: str
    format: str
    dtype: str
    chunk_tokens: int
    chunk_overlap: int
    eos_id: int
    bos_id: int
    vocab_cap: int
    key_bytes: int
    max_eos_drop_rate: float
    hash_seed: bytes | None  # None = the lock phase derives and pins the seed
    include_rules: tuple[str, ...]
    exclude_rules: tuple[str, ...]
    validation_fraction: float
    validation_domain: str
    transforms: tuple[TransformSpec, ...]
    source: dict[str, Any]
    tokenizer_name: str
    tokenizer_kind: str
    tokenizer_rev: str | None  # None = the lock phase resolves the default branch
    publish_root: str | None
    outputs: tuple[OutputSpec, ...]
    raw: dict[str, Any]

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.raw)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise BuildError(f"recipe invalid: {msg}")


def _pos_int(raw: Any, field: str) -> int:
    _require(isinstance(raw, int) and not isinstance(raw, bool) and raw > 0, f"{field} must be a positive int")
    return raw


def _is_hex64(v: Any) -> bool:
    """A real 64-hex digest string — never a length-checked arbitrary string."""
    return isinstance(v, str) and len(v) == 64 and _is_hex(v)


def _is_hex(v: Any) -> bool:
    return isinstance(v, str) and all(c in "0123456789abcdef" for c in v)


def _check_geometry(format: str, chunk_tokens: int, where: str) -> None:
    _require(format in FORMATS, f"{where}: unknown format {format!r}")
    if format == PITHOS_STREAM_V1:
        _require(
            chunk_tokens == PITHOS_CHUNK_TOKENS,
            f"{where}: recipes must build {PITHOS_STREAM_V1} at the approved default "
            f"2**26 logical tokens, got {chunk_tokens}",
        )


def load_recipe(path: str) -> Recipe:
    """Load and fully validate a recipe. Raises BuildError on any violation;
    nothing is resolved or downloaded here."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise BuildError(f"cannot read recipe {path}: {e}") from e
    return recipe_from_raw(raw)


def recipe_from_raw(raw: dict[str, Any]) -> Recipe:
    """Validate a recipe dict (e.g. the copy embedded in a build plan)."""
    if not isinstance(raw, dict):
        raise BuildError("recipe invalid: top level must be an object")
    if raw.get("entry_type") != "recipe":
        raise BuildError("recipe invalid: entry_type must be 'recipe'")
    name = raw.get("recipe")
    if not isinstance(name, str) or not name:
        raise BuildError("recipe invalid: recipe name must be a non-empty string")

    fmt = raw.get("format")
    if not isinstance(fmt, str):
        raise BuildError("recipe invalid: format must be a string")
    chunk_tokens = _pos_int(raw.get("chunk_tokens"), "chunk_tokens")
    _check_geometry(fmt, chunk_tokens, "recipe")
    if raw.get("dtype") not in {"<i4", "int32"}:
        raise BuildError("recipe invalid: dtype must be '<i4' or the legacy 'int32' tag")
    if raw.get("chunk_overlap") != 1:
        raise BuildError("recipe invalid: chunk_overlap must be exactly 1")
    eos_id = _pos_int(raw.get("eos_id"), "eos_id")
    bos_id = _pos_int(raw.get("bos_id"), "bos_id")
    vocab_cap = _pos_int(raw.get("vocab_cap"), "vocab_cap")
    if eos_id >= vocab_cap or bos_id >= vocab_cap:
        raise BuildError("recipe invalid: eos_id/bos_id must be < vocab_cap")
    if raw.get("key_bytes") != 16:
        raise BuildError("recipe invalid: key_bytes must be 16 (BLAKE2b-128)")
    drop = raw.get("max_eos_drop_rate")
    if isinstance(drop, bool) or not isinstance(drop, (int, float)) or not 0.0 <= drop < 1.0:
        raise BuildError("recipe invalid: max_eos_drop_rate must be in [0, 1)")

    seed_raw = raw.get("hash_seed")
    if seed_raw is None:
        hash_seed = None
    else:
        if not isinstance(seed_raw, str):
            raise BuildError("recipe invalid: hash_seed must be null or a hex string")
        try:
            hash_seed = bytes.fromhex(seed_raw)
        except ValueError:
            raise BuildError("recipe invalid: hash_seed is not hex") from None
        if len(hash_seed) != 16:
            raise BuildError("recipe invalid: hash_seed must decode to 16 bytes")

    include = raw.get("include_rules")
    if not isinstance(include, list) or not include or not all(isinstance(r, str) and r for r in include):
        raise BuildError("recipe invalid: include_rules must be a non-empty list of patterns")
    exclude = raw.get("exclude_rules", [])
    if not isinstance(exclude, list) or not all(isinstance(r, str) and r for r in exclude):
        raise BuildError("recipe invalid: exclude_rules must be a list of patterns")

    val = raw.get("validation_reservation")
    if not isinstance(val, dict):
        raise BuildError("recipe invalid: validation_reservation must be an object")
    frac = val.get("fraction")
    if isinstance(frac, bool) or not isinstance(frac, (int, float)) or not 0.0 < frac < 1.0:
        raise BuildError("recipe invalid: validation fraction must be in (0, 1)")
    domain = val.get("domain")
    if not isinstance(domain, str) or not domain:
        raise BuildError("recipe invalid: validation domain must be a non-empty string")

    transforms = resolve_transforms(raw.get("transforms", []))

    source = raw.get("source")
    if not isinstance(source, dict):
        raise BuildError("recipe invalid: source must be an object")
    kind = source.get("kind")
    if kind not in SOURCE_KINDS:
        raise BuildError(f"recipe invalid: source.kind must be one of {sorted(SOURCE_KINDS)}")
    if kind == "huggingface":
        if not isinstance(source.get("dataset"), str) or not source["dataset"]:
            raise BuildError("recipe invalid: huggingface source needs dataset")
    elif kind in {"local", "s3"}:
        if not isinstance(source.get("uri"), str) or not source["uri"]:
            raise BuildError(f"recipe invalid: {kind} source needs a uri")
    elif kind == "legacy_binary":
        # Byte-preserving re-chunk of an existing flat-token stream. The URI
        # and ordered object inventory are REQUIRED lock inputs — the recipe
        # may leave uri null, and locking fails clearly until it is supplied.
        if source.get("uri") is not None and not isinstance(source["uri"], str):
            raise BuildError("recipe invalid: legacy_binary uri must be null (lock-resolved) or a string")
        _pos_int(source.get("legacy_chunk_tokens"), "source legacy_chunk_tokens")
        expected = source.get("expected_logical_tokens")
        if expected is not None:
            _pos_int(expected, "source expected_logical_tokens")
    else:  # http
        if not isinstance(source.get("base_url"), str) or not source["base_url"]:
            raise BuildError("recipe invalid: http source needs base_url")
        check_http_base_url(source["base_url"])
        if not isinstance(source.get("items"), list) or not source["items"]:
            raise BuildError("recipe invalid: http source needs an explicit items list (origins are not listable)")
        check_inventory_names(source["items"], "recipe invalid: http source")

    tok_kind = raw.get("tokenizer_kind")
    if not isinstance(tok_kind, str) or tok_kind not in TOKENIZER_KINDS:
        raise BuildError(f"recipe invalid: tokenizer_kind must be one of {sorted(TOKENIZER_KINDS)}")
    tok_name = raw.get("tokenizer")
    if not isinstance(tok_name, str) or not tok_name:
        raise BuildError("recipe invalid: tokenizer must be a non-empty string")
    tok_rev = raw.get("tokenizer_rev")
    if tok_rev is not None and not isinstance(tok_rev, str):
        raise BuildError("recipe invalid: tokenizer_rev must be null or a revision string")

    publish_root = raw.get("publish_root")
    if publish_root is not None and not isinstance(publish_root, str):
        raise BuildError("recipe invalid: publish_root must be null or a string")

    outputs_raw = raw.get("outputs")
    if not isinstance(outputs_raw, dict) or not outputs_raw:
        raise BuildError("recipe invalid: outputs must be a non-empty object")
    outputs: list[OutputSpec] = []
    for oname, o in sorted(outputs_raw.items()):
        if not isinstance(o, dict):
            raise BuildError(f"recipe invalid: output {oname!r} must be an object")
        if not oname.replace("-", "_").replace(".", "_").isalnum():
            raise BuildError(f"recipe invalid: output name {oname!r} is unsafe")
        role = o.get("role", "train")
        if role not in {"train", "validation"}:
            raise BuildError(f"recipe invalid: output {oname!r}: role must be train|validation")
        ofmt = o.get("format", fmt)
        if not isinstance(ofmt, str):
            raise BuildError(f"recipe invalid: output {oname!r}: format must be a string")
        ochunk = _pos_int(o.get("chunk_tokens", chunk_tokens), f"output {oname!r} chunk_tokens")
        _check_geometry(ofmt, ochunk, f"output {oname!r}")
        budget = o.get("token_budget")
        if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0):
            raise BuildError(f"recipe invalid: output {oname!r}: token_budget must be null or a positive int")
        outputs.append(
            OutputSpec(
                oname, role, ofmt, ochunk, budget, bool(o.get("optional", False)), str(o.get("description", ""))
            )
        )
    if not any(o.role == "train" for o in outputs):
        raise BuildError("recipe invalid: at least one train output is required")

    return Recipe(
        name=name,
        format=fmt,
        dtype=str(raw["dtype"]),
        chunk_tokens=chunk_tokens,
        chunk_overlap=1,
        eos_id=eos_id,
        bos_id=bos_id,
        vocab_cap=vocab_cap,
        key_bytes=16,
        max_eos_drop_rate=float(drop),
        hash_seed=hash_seed,
        include_rules=tuple(include),
        exclude_rules=tuple(exclude),
        validation_fraction=float(frac),
        validation_domain=domain,
        transforms=transforms,
        source=dict(source),
        tokenizer_name=tok_name,
        tokenizer_kind=tok_kind,
        tokenizer_rev=tok_rev,
        publish_root=publish_root,
        outputs=tuple(outputs),
        raw=raw,
    )


@dataclass(frozen=True)
class Lock:
    """The immutable build lock: every pin the build will check, resolved.
    source_lock always carries a per-object inventory with content digests
    ({"name", "size", "sha256"} per item; legacy_binary items add "tokens"
    and are kept in locked ORDER); tokenizer_lock pins the asset digest,
    implementation version, and the exact BOS/EOS/add-special-tokens policy;
    code is pinned by VCS revision AND the pithos source-tree digest."""

    recipe_sha256: str
    source_lock: dict[str, Any]
    tokenizer_lock: dict[str, Any]
    code_revision: str
    code_tree_sha256: str
    transforms_sha256: str
    hash_seed_hex: str
    publish_root: str | None

    @property
    def source_sha256(self) -> str:
        return _canonical_sha256(self.source_lock)

    @property
    def tokenizer_sha256(self) -> str:
        return _canonical_sha256(self.tokenizer_lock)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lock_version": LOCK_VERSION,
            "recipe_sha256": self.recipe_sha256,
            "source_lock": self.source_lock,
            "tokenizer_lock": self.tokenizer_lock,
            "code_revision": self.code_revision,
            "code_tree_sha256": self.code_tree_sha256,
            "transforms_sha256": self.transforms_sha256,
            "hash_seed": self.hash_seed_hex,
            "publish_root": self.publish_root,
        }

    @property
    def hash_seed(self) -> bytes:
        return bytes.fromhex(self.hash_seed_hex)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Lock:
        """Validate a lock dict (from a lock file or a build plan) strictly:
        real hex digests (not just length), per-item size/shape, kind-
        specific fields, and typed tokenizer/code pins — a malformed plan or
        lock never reaches a KeyError or an oversized operation downstream."""
        if not isinstance(raw, dict):
            raise BuildError("lock is malformed: top level must be an object")
        try:
            if raw["lock_version"] != LOCK_VERSION:
                raise BuildError("unsupported lock_version")
            if not _is_hex64(raw.get("recipe_sha256")):
                raise BuildError("bad recipe_sha256 (want 64 hex)")
            src = raw["source_lock"]
            if not isinstance(src, dict):
                raise BuildError("bad source_lock")
            kind = src.get("kind")
            if kind not in SOURCE_KINDS:
                raise BuildError(f"bad source_lock: kind must be one of {sorted(SOURCE_KINDS)}")
            if kind == "huggingface" and (not isinstance(src.get("dataset"), str) or not src["dataset"]):
                raise BuildError("bad source_lock: huggingface needs a dataset")
            if kind == "huggingface" and (not isinstance(src.get("revision"), str) or not src["revision"]):
                raise BuildError("bad source_lock: huggingface needs a pinned revision")
            if kind == "http" and (not isinstance(src.get("base_url"), str) or not src["base_url"]):
                raise BuildError("bad source_lock: http needs a base_url")
            if kind in {"local", "s3", "legacy_binary"} and (not isinstance(src.get("uri"), str) or not src["uri"]):
                raise BuildError(f"bad source_lock: {kind} needs a uri")
            if kind == "legacy_binary":
                c = src.get("legacy_chunk_tokens")
                if not isinstance(c, int) or isinstance(c, bool) or c <= 0:
                    raise BuildError("bad source_lock: legacy_chunk_tokens must be a positive int")
                expected = src.get("expected_logical_tokens")
                if expected is not None and (
                    not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0
                ):
                    raise BuildError("bad source_lock: expected_logical_tokens must be null or a positive int")
            items = src.get("items")
            if not isinstance(items, list):
                raise BuildError("bad source_lock: items must be a list")
            seen_names: set[str] = set()
            for i in items:
                if not isinstance(i, dict):
                    raise BuildError("bad source_lock: every item must be an object")
                check_object_name(i.get("name"), "bad source_lock: item")
                if i["name"] in seen_names:
                    raise BuildError(f"bad source_lock: duplicate item name {i['name']!r}")
                seen_names.add(i["name"])
                if not _is_hex64(i.get("sha256")):
                    raise BuildError(f"bad source_lock: item {i['name']!r} sha256 must be 64 hex")
                if not isinstance(i.get("size"), int) or isinstance(i["size"], bool) or i["size"] < 0:
                    raise BuildError(f"bad source_lock: item {i['name']!r} size must be a non-negative int")
                if kind == "legacy_binary":
                    t = i.get("tokens")
                    if not isinstance(t, int) or isinstance(t, bool) or t <= 0:
                        raise BuildError(f"bad source_lock: legacy item {i['name']!r} tokens must be a positive int")
            tok = raw["tokenizer_lock"]
            if (
                not isinstance(tok, dict)
                or not isinstance(tok.get("name"), str)
                or not tok["name"]
                or not isinstance(tok.get("revision"), str)
                or not tok["revision"]
                or not _is_hex64(tok.get("asset_sha256"))
                or not isinstance(tok.get("impl_version"), str)
                or not tok["impl_version"]
                or not isinstance(tok.get("bos_id"), int)
                or isinstance(tok["bos_id"], bool)
                or tok["bos_id"] < 0
                or not isinstance(tok.get("eos_id"), int)
                or isinstance(tok["eos_id"], bool)
                or tok["eos_id"] < 0
                or not isinstance(tok.get("add_special_tokens"), bool)
                or tok["add_special_tokens"] is not False
            ):
                raise BuildError("bad tokenizer_lock")
            if not isinstance(raw["code_revision"], str) or not raw["code_revision"]:
                raise BuildError("bad code_revision")
            if not _is_hex64(raw.get("code_tree_sha256")):
                raise BuildError("bad code_tree_sha256 (want 64 hex)")
            if not _is_hex64(raw.get("transforms_sha256")):
                raise BuildError("bad transforms_sha256 (want 64 hex)")
            seed = raw["hash_seed"]
            if not isinstance(seed, str) or len(seed) != 32 or not _is_hex(seed):
                raise BuildError("bad hash_seed (want 16 bytes hex)")
            if raw.get("publish_root") is not None and not isinstance(raw["publish_root"], str):
                raise BuildError("bad publish_root")
        except (KeyError, ValueError) as e:
            raise BuildError(f"lock is malformed: {e}") from e
        return cls(
            recipe_sha256=raw["recipe_sha256"],
            source_lock=src,
            tokenizer_lock=tok,
            code_revision=raw["code_revision"],
            code_tree_sha256=raw["code_tree_sha256"],
            transforms_sha256=raw["transforms_sha256"],
            hash_seed_hex=raw["hash_seed"],
            publish_root=raw.get("publish_root"),
        )


def _check_items(items: list[dict[str, Any]], what: str) -> None:
    if not items:
        raise BuildError(f"lock failed: {what} inventory was not resolved (no items)")
    for i in items:
        if not isinstance(i, dict):
            raise BuildError(f"lock failed: {what} item must be an object")
        check_object_name(i.get("name"), f"lock failed: {what} item")
        if not _is_hex64(i.get("sha256")):
            raise BuildError(
                f"lock failed: {what} item {i.get('name')!r} has no content sha256 — never pin a path as a digest"
            )
        if not isinstance(i.get("size"), int) or isinstance(i["size"], bool) or i["size"] < 0:
            raise BuildError(f"lock failed: {what} item {i['name']!r} has no size")
    check_inventory_names([i["name"] for i in items], f"lock failed: {what}")


def lock_recipe(
    recipe: Recipe,
    *,
    source_revision: str | None = None,
    source_items: list[dict[str, Any]] | None = None,
    tokenizer_evidence: dict[str, str],
    code_revision: str,
    code_tree_sha256: str,
    seed_bytes: bytes | None = None,
    publish_root: str | None = None,
    source_uri: str | None = None,
) -> Lock:
    """Resolve every lock-phase pin. The caller supplies the resolved
    evidence (this function never touches the network itself); anything
    unresolvable must fail in the resolver BEFORE we get here. Raises
    BuildError on missing/invalid pins — nothing is invented silently.

    source_items: per-object [{"name", "size", "sha256"}] evidence for the
    SELECTED inventory (post include/exclude). legacy_binary additionally
    requires per-item "tokens" and preserves the given order (it is the
    stream order); all other kinds are sorted by name.

    publish_root / source_uri: lock-time CLI inputs for recipes that leave
    them null. When the recipe already pins one, a differing explicit value
    is a loud error — the lock never picks silently between two values."""
    kind = recipe.source["kind"]
    if kind == "huggingface":
        if not isinstance(source_revision, str) or not source_revision:
            raise BuildError("lock failed: huggingface source revision was not resolved")
        _check_items(source_items or [], "huggingface")
        source_lock: dict[str, Any] = {
            "kind": kind,
            "dataset": recipe.source["dataset"],
            "revision": source_revision,
            "items": sorted(source_items or [], key=lambda i: i["name"]),
        }
    elif kind == "legacy_binary":
        recipe_uri = recipe.source.get("uri")
        if recipe_uri is not None and source_uri is not None and source_uri != recipe_uri:
            raise BuildError(f"lock failed: explicit source uri {source_uri!r} != recipe's pinned {recipe_uri!r}")
        uri = recipe_uri if isinstance(recipe_uri, str) and recipe_uri else source_uri
        if not isinstance(uri, str) or not uri:
            raise BuildError(
                "lock failed: legacy_binary source has no uri — supply the locked legacy stream location explicitly"
            )
        _check_items(source_items or [], "legacy_binary")
        for i in source_items or []:
            if not isinstance(i.get("tokens"), int) or isinstance(i["tokens"], bool) or i["tokens"] <= 0:
                raise BuildError(f"lock failed: legacy_binary item {i['name']!r} has no token count")
        source_lock = {
            "kind": kind,
            "uri": uri,
            "legacy_chunk_tokens": recipe.source["legacy_chunk_tokens"],
            "expected_logical_tokens": recipe.source.get("expected_logical_tokens"),
            "items": list(source_items or []),  # locked ORDER is the stream order
        }
    else:
        _check_items(source_items or [], kind)
        base = {"kind": kind}
        if kind == "http":
            base["base_url"] = recipe.source["base_url"]
        else:
            base["uri"] = recipe.source["uri"]
        source_lock = {**base, "items": sorted(source_items or [], key=lambda i: i["name"])}

    if not isinstance(tokenizer_evidence, dict):
        raise BuildError("lock failed: tokenizer evidence was not resolved")
    trev = tokenizer_evidence.get("revision")
    tasset = tokenizer_evidence.get("asset_sha256")
    timpl = tokenizer_evidence.get("impl_version")
    if not isinstance(trev, str) or not trev:
        raise BuildError("lock failed: tokenizer revision was not resolved")
    if not _is_hex64(tasset):
        raise BuildError("lock failed: tokenizer asset sha256 was not resolved")
    if not isinstance(timpl, str) or not timpl:
        raise BuildError("lock failed: tokenizer implementation version was not resolved")
    if not isinstance(code_revision, str) or not code_revision:
        raise BuildError("lock failed: code revision was not resolved")
    if not _is_hex64(code_tree_sha256):
        raise BuildError("lock failed: pithos source-tree digest was not resolved")

    if recipe.publish_root is not None and publish_root is not None and publish_root != recipe.publish_root:
        raise BuildError(
            f"lock failed: explicit publish root {publish_root!r} != recipe's pinned {recipe.publish_root!r}"
        )
    locked_publish_root = recipe.publish_root if recipe.publish_root is not None else publish_root

    if recipe.hash_seed is not None:
        seed = recipe.hash_seed
    elif seed_bytes is not None:
        if len(seed_bytes) != 16:
            raise BuildError("lock failed: explicit seed must be 16 bytes")
        seed = seed_bytes
    else:
        # Per-corpus seed, derived deterministically from the recipe and the
        # locked source state — unique to this corpus, reproducible on
        # re-lock, and never the legacy 'colonnade-165' pin.
        seed = hashlib.blake2b(
            bytes.fromhex(recipe.sha256) + _canonical_sha256(source_lock).encode(), digest_size=16
        ).digest()

    return Lock(
        recipe_sha256=recipe.sha256,
        source_lock=source_lock,
        tokenizer_lock={
            "name": recipe.tokenizer_name,
            "revision": trev,
            "asset_sha256": tasset,
            "impl_version": timpl,
            "bos_id": recipe.bos_id,
            "eos_id": recipe.eos_id,
            "add_special_tokens": False,
        },
        code_revision=code_revision,
        code_tree_sha256=code_tree_sha256,
        transforms_sha256=transform_identity_digest(recipe.transforms),
        hash_seed_hex=seed.hex(),
        publish_root=locked_publish_root,
    )


def lock_path_for(recipe_path: str) -> str:
    base, ext = os.path.splitext(recipe_path)
    return f"{base}.lock{ext or '.json'}"


def write_lock(lock: Lock, path: str) -> bool:
    """ATOMIC create-once lock publication: the serialized lock is written
    to a PRIVATE same-directory temp file, flushed and fsynced, then
    published with a single atomic hard-link that fails if the destination
    exists. The destination is never visible empty or partial — not to
    readers, not to concurrent lockers — and a crash leaves only the
    reclaimable temp file, never a corrupt lock. Returns False when an
    identical lock already exists (byte-verified), raises BuildError on a
    DIFFERENT existing lock."""
    data = (json.dumps(lock.to_dict(), indent=2, sort_keys=True) + "\n").encode()
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), prefix=".lock-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)  # atomic create-once: fails if the destination exists
        except FileExistsError:
            with open(path, "rb") as f:
                if f.read() == data:
                    return False
            raise BuildError(f"lock {path} already exists with different content — locks are immutable") from None
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return True


def load_lock(path: str) -> Lock:
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise BuildError(f"cannot read lock {path}: {e}") from e
    try:
        return Lock.from_dict(raw)
    except BuildError as e:
        raise BuildError(f"lock {path}: {e}") from e
