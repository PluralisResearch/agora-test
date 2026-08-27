"""The deterministic build engine: plan → work → finalize → verify → publish.

Determinism is structural, not incidental:

* Work units are the selected source items in locked (sorted) order; worker
  N of M claims units at positions ≡ N (mod M). Assignment affects only
  WHICH part file a record lands in — never the final bytes, because
  finalize performs a global keyed merge over ALL parts and chunking uses
  fixed stream positions (strict 2**26 + 1 overlap for
  pithos_stream_v1). Output is byte-identical across worker counts and
  retry assignments by construction, and markers pin no worker identity.
* Memory and bandwidth are bounded at every phase: the plan scan reads row
  evidence via metadata-only Parquet FOOTER reads (pinned-revision ranged
  reads for HF/S3 — never a full-object download, resumable through
  create-once per-unit scan state); workers spool ONE source object at a
  time to a collision-free local file, reclaim it before the next unit, and
  never use the HF hub cache for source data; records are block-sorted and
  spilled to local run files; every merge (unit runs AND final parts) is a
  bounded-fan-in iterative k-way merge whose intermediate runs are
  reclaimed on success and failure; chunks are emitted one at a time
  straight to storage; all hashing is incremental. Nothing holds a
  corpus-sized or object-set-sized list of bytes.
* Resume is marker-gated: a unit with a valid completion marker (all lock
  pins match, every part checksum verifies) is skipped; anything else is
  rebuilt deterministically. Publication is atomic create-once at every
  level: a name that already exists is verified byte-identical (idempotent
  retry, including mid-phase crashes and concurrent duplicate workers) or
  rejected loudly — never silently overwritten.

This subsumes the legacy fineweb_shuffle_reshard pipeline: the keyed global
order replaces the approximate/full per-dump shuffles and the
producer/consumer merge-reshard (with no trainer-count-dependent shard
ranges and no imbalance), and marker/manifest reconciliation replaces the
doc-count verification scripts. The legacy dev repetition filter and global
dedup are deliberately NOT imported.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import shutil
import sys
import tempfile

from collections.abc import Callable, Generator, Iterator
from contextlib import closing
from typing import Any, BinaryIO

import numpy as np

from ..corpus import ChunkWriter, merge_parts, read_records, write_record
from ..errors import BuildError
from ..manifest import Manifest, manifest_sha256
from . import markers as markers_mod
from . import policy as pol
from .codepin import pithos_tree_sha256
from .recipe import Lock, OutputSpec, Recipe, _is_hex64, recipe_from_raw
from .sources import (
    INVENTORY_SUFFIXES,
    HfHubClient,
    ItemHandle,
    SourceItem,
    adapter_for,
    declared_rows,
    hf_declared_rows,
    items_from_lock,
    open_verified_local,
    spool_stream,
)
from .storage import ImmutableConflict, LocalStorage, Storage, storage_for
from .tokenize import TokenizerPolicy, check_eos_drop_rate, encode_stream, tokenizer_for


PLAN_VERSION = 2
RUN_BLOCK_RECORDS = 32768  # bound on records held in memory per sort block
MERGE_FANIN = 128  # bound on simultaneously open streams in every merge
LEGACY_SLICE_TOKENS = 1 << 20  # bound on tokens per write in the legacy rechunk path


def _json_bytes(d: dict[str, Any]) -> bytes:
    return (json.dumps(d, indent=2, sort_keys=True) + "\n").encode()


def _enforce_code_tree(lock: Lock) -> None:
    """Every engine phase — plan, worker, finalize, verify, publish — first
    proves the RUNNING pithos source tree matches the locked digest before
    any source is listed/downloaded or any byte is produced, merged,
    certified, or published. Otherwise a code change after the lock would
    emit parts/chunks whose markers falsely claim the old tree."""
    current = pithos_tree_sha256()
    if current != lock.code_tree_sha256:
        raise BuildError(
            f"the running pithos source tree ({current[:16]}…) != the locked tree "
            f"({lock.code_tree_sha256[:16]}…) — the code changed after the lock; re-lock under the current tree"
        )


def _write_immutable(storage: Storage, name: str, data: bytes) -> bool:
    """Atomic create-once publication with idempotent retry: write if absent
    or byte-identical (returns False); a DIFFERENT existing object fails.
    The create-once is atomic in the storage layer — no check-then-act."""
    try:
        storage.write_bytes(name, data, create_once=True)
        return True
    except ImmutableConflict:
        if storage.read_bytes(name) == data:
            return False
        raise BuildError(
            f"{name!r} already exists with different content — immutable artifacts are never overwritten"
        ) from None


def _load_plan(build: Storage) -> dict[str, Any]:
    if not build.exists("plan.json"):
        raise BuildError("build has no plan.json — run 'pithos build plan' first")
    try:
        plan = json.loads(build.read_bytes("plan.json"))
    except json.JSONDecodeError as e:
        raise BuildError(f"plan.json is corrupt: {e}") from e
    if not isinstance(plan, dict) or plan.get("plan_version") != PLAN_VERSION:
        raise BuildError("plan.json: unsupported plan_version")
    if not isinstance(plan.get("recipe"), dict) or not isinstance(plan.get("lock"), dict):
        raise BuildError("plan.json: missing recipe or lock")
    if not isinstance(plan.get("units"), list) or not isinstance(plan.get("outputs"), list):
        raise BuildError("plan.json: missing units or outputs")
    for u in plan["units"]:
        if not isinstance(u, dict) or not isinstance(u.get("unit_id"), str) or not u["unit_id"]:
            raise BuildError("plan.json: malformed unit (unit_id)")
        if not isinstance(u.get("crawl"), str):
            raise BuildError("plan.json: malformed unit (crawl)")
        item = u.get("item")
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not item["name"]
            or not _is_hex64(item.get("sha256"))
            or not isinstance(item.get("size"), int)
            or isinstance(item["size"], bool)
            or item["size"] < 0
        ):
            raise BuildError("plan.json: malformed unit item (name/size/sha256)")
        rows = u.get("rows")
        if rows is not None and (not isinstance(rows, int) or isinstance(rows, bool) or rows < 0):
            raise BuildError("plan.json: malformed unit rows evidence")
    return plan


def _plan_recipe_lock(plan: dict[str, Any]) -> tuple[Recipe, Lock]:
    recipe = recipe_from_raw(plan["recipe"])
    lock = Lock.from_dict(plan["lock"])
    if lock.recipe_sha256 != recipe.sha256:
        raise BuildError("plan.json: lock does not match the embedded recipe")
    return recipe, lock


def _plan_outputs(plan: dict[str, Any]) -> list[OutputSpec]:
    outputs: list[OutputSpec] = []
    for o in plan["outputs"]:
        if not isinstance(o, dict):
            raise BuildError("plan.json: malformed output payload")
        name, role, fmt = o.get("name"), o.get("role"), o.get("format")
        chunk_tokens, budget, optional = o.get("chunk_tokens"), o.get("token_budget"), o.get("optional")
        if not isinstance(name, str) or not name or role not in {"train", "validation"} or not isinstance(fmt, str):
            raise BuildError("plan.json: malformed output payload (name/role/format)")
        if not isinstance(chunk_tokens, int) or isinstance(chunk_tokens, bool) or chunk_tokens <= 0:
            raise BuildError(f"plan.json: output {name!r} chunk_tokens must be a positive int")
        if budget is not None and (not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0):
            raise BuildError(f"plan.json: output {name!r} token_budget must be null or a positive int")
        outputs.append(OutputSpec(name, role, fmt, chunk_tokens, budget, bool(optional), ""))
    return outputs


class _Hasher:
    """write() shim that hashes everything written to a stream."""

    def __init__(self, f: BinaryIO) -> None:
        self.f = f
        self.h = hashlib.sha256()

    def write(self, b: bytes) -> int:
        self.h.update(b)
        return self.f.write(b)


# ------------------------------------------------------------------- plan
def plan_build(
    recipe: Recipe,
    lock: Lock,
    build: Storage,
    *,
    client: Any = None,
    fetcher: Any = None,
) -> dict[str, Any]:
    """Publish plan.json (create-once) from the LOCKED source inventory —
    the lock is authoritative; no live re-listing decides the build. For
    local/s3 sources the live listing's (name, size) must still match the
    lock under the SAME inventory rule the CLI locked with (the
    INVENTORY_SUFFIXES filter); harmless non-source files are ignored in
    both places. Parquet row evidence is captured per unit into create-once
    scan state (metadata-only footer reads — never full objects), so the
    scan is resumable across thousands of ranged requests and the plan's
    row pins are independent of any worker-derived count."""
    _enforce_code_tree(lock)
    if lock.recipe_sha256 != recipe.sha256:
        raise BuildError("the lock does not match this recipe — re-run 'pithos recipe lock'")
    kind = recipe.source["kind"]
    items = items_from_lock(lock.source_lock)

    if kind == "legacy_binary":
        plan = {
            "plan_version": PLAN_VERSION,
            "recipe": recipe.raw,
            "lock": lock.to_dict(),
            "units": [],
            "outputs": _output_payloads(recipe),
        }
        _write_immutable(build, "plan.json", _json_bytes(plan))
        return plan

    scanner: Callable[[SourceItem], int | None]
    if kind in {"local", "s3"}:
        live_storage = storage_for(recipe.source["uri"], client=client)
        live = sorted(
            ((o.name, o.size) for o in live_storage.list("") if o.name.endswith(INVENTORY_SUFFIXES)),
            key=lambda t: t[0],
        )
        locked = sorted(((i.name, i.size) for i in items), key=lambda t: t[0])
        if live != locked:
            raise BuildError(
                "live source inventory differs from the locked listing — the source changed after the lock"
            )

        def scan_via_storage(i: SourceItem, _s: Storage = live_storage) -> int | None:
            return declared_rows(_s, i)

        scanner = scan_via_storage
    elif kind == "huggingface":
        hub = client if client is not None else HfHubClient()
        dataset, revision = recipe.source["dataset"], str(lock.source_lock["revision"])

        def scan_via_hf(i: SourceItem) -> int | None:
            return hf_declared_rows(hub, dataset, revision, i)

        scanner = scan_via_hf
    else:  # http: recipe-declared items carry no footer row evidence

        def scan_none(_i: SourceItem) -> int | None:
            return None

        scanner = scan_none

    selected = [i for i in items if pol.selected(i.name, recipe.include_rules, recipe.exclude_rules)]
    if not selected:
        raise BuildError("selection is empty — no source items match the recipe's include/exclude rules")

    units = []
    for pos, i in enumerate(selected):
        unit_id = f"unit-{pos:05d}"
        units.append(
            {
                "unit_id": unit_id,
                "item": {"name": i.name, "size": i.size, "sha256": i.sha256},
                "crawl": pol.crawl_of(i.name),
                "rows": _unit_rows(build, unit_id, i, scanner),
            }
        )
    plan = {
        "plan_version": PLAN_VERSION,
        "recipe": recipe.raw,
        "lock": lock.to_dict(),
        "units": units,
        "outputs": _output_payloads(recipe),
    }
    _write_immutable(build, "plan.json", _json_bytes(plan))
    return plan


def _unit_rows(
    build: Storage, unit_id: str, item: SourceItem, scanner: Callable[[SourceItem], int | None]
) -> int | None:
    """Row evidence for one unit, pinned into create-once scan state: an
    existing scan file is validated against the lock and reused (resumable
    scan); otherwise the ranged scanner runs once and the result is
    published atomically. Worker counts never feed this evidence."""
    name = f"scan/{unit_id}.json"
    item_payload = {"name": item.name, "size": item.size, "sha256": item.sha256}
    if build.exists(name):
        try:
            payload = json.loads(build.read_bytes(name))
        except json.JSONDecodeError as e:
            raise BuildError(f"{name} is corrupt: {e}") from e
        if payload.get("unit_id") != unit_id or payload.get("item") != item_payload:
            raise BuildError(f"{name} does not match the locked {unit_id} — refusing to reuse inconsistent scan state")
        rows = payload.get("rows")
        if rows is not None and (not isinstance(rows, int) or isinstance(rows, bool) or rows < 0):
            raise BuildError(f"{name}: malformed rows evidence")
        return rows
    rows = scanner(item)
    _write_immutable(build, name, _json_bytes({"unit_id": unit_id, "item": item_payload, "rows": rows}))
    return rows


def _output_payloads(recipe: Recipe) -> list[dict[str, Any]]:
    return [
        {
            "name": o.name,
            "role": o.role,
            "format": o.format,
            "chunk_tokens": o.chunk_tokens,
            "token_budget": o.token_budget,
            "optional": o.optional,
        }
        for o in recipe.outputs
    ]


# ------------------------------------------------------------------ worker
def run_worker(
    build: Storage,
    worker_index: int,
    worker_count: int,
    *,
    tokenizer: TokenizerPolicy | None = None,
    client: Any = None,
    fetcher: Any = None,
    spool_dir: str | None = None,
    run_block_records: int = RUN_BLOCK_RECORDS,
    merge_fanin: int = MERGE_FANIN,
) -> dict[str, Any]:
    """Process this worker's assigned units with bounded memory: open ONE
    verified source object at a time (local files hashed from the parsed
    descriptor itself; remote objects streamed to a collision-free spool),
    encode in bounded batches, block-sort and spill run files, bounded
    fan-in merge runs into create-once parts (one per role), then publish
    the marker. Idempotent: units with valid markers are skipped (checksums
    re-verified); a part that exists from a crashed attempt must hash
    identically. Every scratch file (spooled object, runs, owned spool dir)
    is reclaimed on success AND on failure; a caller-provided spool_dir is
    never deleted."""
    if worker_count < 1 or not 0 <= worker_index < worker_count:
        raise BuildError(f"invalid worker identity {worker_index}/{worker_count}")
    if run_block_records < 1:
        raise BuildError(f"run_block_records {run_block_records} < 1")
    plan = _load_plan(build)
    recipe, lock = _plan_recipe_lock(plan)
    _enforce_code_tree(lock)
    if recipe.source["kind"] == "legacy_binary":
        return {"assigned": 0, "completed": [], "skipped": [], "max_resident_records": 0}
    if tokenizer is None:
        tokenizer = tokenizer_for(recipe, lock.tokenizer_lock, client=client)
    adapter = adapter_for(recipe, lock.source_lock, client=client, fetcher=fetcher)
    roles = sorted({o["role"] for o in plan["outputs"]})

    units = plan["units"]
    assigned = [u for pos, u in enumerate(units) if pos % worker_count == worker_index]
    done: list[str] = []
    skipped: list[str] = []
    max_resident = 0
    for unit in assigned:
        unit_id = unit["unit_id"]
        if build.exists(markers_mod.marker_name(unit_id)):
            markers_mod.load_marker(build, unit_id, lock, unit=unit, roles=roles)
            skipped.append(unit_id)
            continue
        item = SourceItem(unit["item"]["name"], unit["item"]["size"], unit["item"]["sha256"])
        owned_spool = spool_dir is None
        spool = spool_dir if spool_dir is not None else tempfile.mkdtemp(prefix=f"pithos-{unit_id}-")
        runs: list[str] = []
        handle: ItemHandle | None = None
        try:
            handle = adapter.open_item(item, spool)
            stats = {"seen": 0, "selected": 0, "dropped_eos": 0, "validation": 0, "tokens": 0}
            block: list[tuple[bytes, bytes]] = []
            for doc in encode_stream(
                adapter.iter_documents(item, handle),
                tokenizer,
                seed=lock.hash_seed,
                bos_id=recipe.bos_id,
                eos_id=recipe.eos_id,
                vocab_cap=recipe.vocab_cap,
                validation_domain=recipe.validation_domain,
                validation_fraction=recipe.validation_fraction,
                transforms=recipe.transforms,
                stats=stats,
            ):
                block.append((doc.key, doc.token_bytes))
                if len(block) >= run_block_records:
                    _spill_block(block, runs, spool)
            _spill_block(block, runs, spool)
            max_resident = max(max_resident, min(run_block_records, stats["selected"]))
            check_eos_drop_rate(stats, recipe.max_eos_drop_rate)

            role_payload = _write_role_parts(build, unit_id, roles, runs, spool, recipe, merge_fanin)
        finally:
            if handle is not None:
                handle.close()
                if handle.spooled is not None and os.path.exists(handle.spooled):
                    os.unlink(handle.spooled)
            for run_path in runs:
                if os.path.exists(run_path):
                    os.unlink(run_path)
            if owned_spool:
                shutil.rmtree(spool, ignore_errors=True)
        marker = markers_mod.build_marker(
            lock=lock, unit_id=unit_id, item=unit["item"], stats=stats, outputs=role_payload
        )
        markers_mod.write_marker(build, marker)
        done.append(unit_id)
    return {"assigned": len(assigned), "completed": done, "skipped": skipped, "max_resident_records": max_resident}


def _spill_block(block: list[tuple[bytes, bytes]], runs: list[str], spool: str) -> None:
    """Sort one bounded block of records and spill it to a local run file."""
    if not block:
        return
    block.sort(key=lambda r: r[0])
    for a, b in zip(block, block[1:]):
        if a[0] == b[0]:
            raise BuildError(f"duplicate (crawl, document_id) key {a[0].hex()} within one work unit")
    run_path = os.path.join(spool, f"run-{len(runs):05d}.bin")
    with open(run_path, "wb") as f:
        for k, raw in block:
            write_record(f, k, len(raw) // 4, raw)
    runs.append(run_path)
    block.clear()


def _merge_run_streams(
    runs: list[str], spool_dir: str, fanin: int = MERGE_FANIN
) -> Generator[tuple[bytes, bytes], None, None]:
    """Bounded fan-in streaming merge of a unit's sorted run files."""
    return _merge_bounded([(p, True) for p in runs], spool_dir=spool_dir, fanin=fanin, what="run")


def _write_role_parts(
    build: Storage,
    unit_id: str,
    roles: list[str],
    runs: list[str],
    spool_dir: str,
    recipe: Recipe,
    fanin: int,
) -> dict[str, dict[str, Any]]:
    """One streaming pass over the unit's merged records, routed by the
    validation reservation into one create-once part per role. Records are
    written as they stream out of the run merge — nothing per-role
    accumulates in memory. A create-once conflict (retry, or a concurrent
    duplicate worker) is verified hash-identical, never overwritten."""
    names = {role: markers_mod.part_name(unit_id, role) for role in roles}
    cms: dict[str, Any] = {}
    hashers: dict[str, _Hasher] = {}
    counts = {role: {"docs": 0, "tokens": 0} for role in roles}
    entered: list[str] = []
    try:
        for role in roles:
            cm = build.open_write(names[role], create_once=True)
            hashers[role] = _Hasher(cm.__enter__())
            cms[role] = cm
            entered.append(role)
        with closing(_merge_run_streams(runs, spool_dir, fanin)) as merged:
            for key, raw in merged:
                role = (
                    "validation"
                    if "validation" in hashers
                    and pol.is_validation(recipe.validation_domain, recipe.validation_fraction, key)
                    else "train"
                )
                write_record(hashers[role], key, len(raw) // 4, raw)
                counts[role]["docs"] += 1
                counts[role]["tokens"] += len(raw) // 4
    except BaseException:
        for role in entered:  # best-effort abort: discard the spooled temp writes
            cms[role].__exit__(*sys.exc_info())
        raise
    conflicts: dict[str, ImmutableConflict] = {}
    for role in entered:
        try:
            cms[role].__exit__(None, None, None)
        except ImmutableConflict as e:
            conflicts[role] = e
    out: dict[str, dict[str, Any]] = {}
    for role in roles:
        sha = hashers[role].h.hexdigest()
        if role in conflicts and build.sha256(names[role]) != sha:
            raise BuildError(
                f"{names[role]!r} already exists with different content — immutable artifacts are never overwritten"
            ) from None
        out[role] = {
            "docs": counts[role]["docs"],
            "tokens": counts[role]["tokens"],
            "part": names[role],
            "part_sha256": sha,
        }
    return out


# ------------------------------------------------------- streaming merges
def _open_merge_inputs(build: Storage | None, inputs: list[tuple[str, bool]]) -> list[BinaryIO]:
    files: list[BinaryIO] = []
    for name, local in inputs:
        if local:
            files.append(open(name, "rb"))
        else:
            if build is None:
                raise BuildError("a storage merge input needs the build storage")
            files.append(build.open_read(name))
    return files


def _kway(files: list[BinaryIO], what: str) -> Iterator[tuple[bytes, bytes]]:
    """One k-way merge level over open record streams with loud
    order/duplicate checks. One record resident at a time per stream."""
    last: bytes | None = None
    for key, raw in heapq.merge(*[read_records(f) for f in files], key=lambda r: r[0]):
        if last is not None:
            if key == last:
                raise BuildError(f"duplicate 128-bit key {key.hex()} across {what}s — investigate")
            if key < last:
                raise BuildError(f"{what} merge order violation — corrupt {what} file")
        last = key
        yield key, raw


def _merge_bounded(
    inputs: list[tuple[str, bool]],
    *,
    build: Storage | None = None,
    spool_dir: str,
    fanin: int,
    what: str,
) -> Generator[tuple[bytes, bytes], None, None]:
    """Bounded-fan-in iterative k-way merge. inputs are (name, is_local):
    storage objects are read via `build`, local run files via open() — an
    explicit flag, never a path-shape guess. When the input count exceeds
    the fan-in, groups of at most `fanin` are merged into deterministic
    intermediate local runs first (open-stream count stays bounded for any
    input size, including > fanin**2), and EVERY intermediate run is
    reclaimed on success and on error."""
    if fanin < 2:
        raise BuildError(f"merge fan-in {fanin} < 2")
    names = list(inputs)
    intermediates: list[str] = []
    try:
        round_no = 0
        while len(names) > fanin:
            nxt: list[tuple[str, bool]] = []
            for g, start in enumerate(range(0, len(names), fanin)):
                group = names[start : start + fanin]
                fd, run_path = tempfile.mkstemp(dir=spool_dir, prefix=f"im-r{round_no}-g{g:05d}-", suffix=".bin")
                os.close(fd)
                intermediates.append(run_path)  # tracked from creation: reclaimed even when this group fails
                files = _open_merge_inputs(build, group)
                try:
                    with open(run_path, "wb") as out:
                        for key, raw in _kway(files, what):
                            write_record(out, key, len(raw) // 4, raw)
                finally:
                    for f in files:
                        f.close()
                nxt.append((run_path, True))  # subsequent rounds read LOCAL intermediates
            names = nxt
            round_no += 1
        files = _open_merge_inputs(build, names)
        try:
            yield from _kway(files, what)
        finally:
            for f in files:
                f.close()
    finally:
        for p in intermediates:
            if os.path.exists(p):
                os.unlink(p)


def _merged_part_stream(
    build: Storage, part_names: list[str], spool_dir: str, fanin: int = MERGE_FANIN
) -> Generator[tuple[bytes, bytes], None, None]:
    """Global streaming k-way merge over all parts (bounded fan-in: many
    parts are merged in deterministic rounds into local intermediate runs
    first, which are reclaimed when the merge closes)."""
    return _merge_bounded([(n, False) for n in part_names], build=build, spool_dir=spool_dir, fanin=fanin, what="part")


# ---------------------------------------------------------------- finalize
def finalize_build(build: Storage, *, merge_fanin: int = MERGE_FANIN) -> dict[str, Any]:
    """Merge all unit parts in global key order into strict-geometry chunks
    (streamed, one chunk at a time, create-once) and publish each output's
    manifest. Reconciles merged counts against the markers. Byte-identical
    for any worker count. The legacy_binary kind takes the byte-preserving
    re-chunk path instead (no keys, no tokenization). The merge scratch
    directory is always reclaimed."""
    plan = _load_plan(build)
    recipe, lock = _plan_recipe_lock(plan)
    _enforce_code_tree(lock)
    if recipe.source["kind"] == "legacy_binary":
        return _finalize_legacy(build, recipe, lock)
    outputs = _plan_outputs(plan)
    roles = sorted({o.role for o in outputs})
    all_markers = [markers_mod.load_marker(build, u["unit_id"], lock, unit=u, roles=roles) for u in plan["units"]]
    spool = tempfile.mkdtemp(prefix="pithos-finalize-")
    try:
        results: dict[str, Any] = {}
        for output in outputs:
            role_parts = [m["outputs"][output.role] for m in all_markers]
            part_names = [p["part"] for p in role_parts]
            declared_docs = sum(p["docs"] for p in role_parts)
            declared_tokens = sum(p["tokens"] for p in role_parts)

            prefix = f"outputs/{output.name}"

            def sink(_idx: int, blob: bytes, _p=prefix) -> None:
                name = f"{_p}/chunk-{_idx:05d}.bin"
                try:
                    build.write_bytes(name, blob, create_once=True)
                except ImmutableConflict:
                    if build.sha256(name) != hashlib.sha256(blob).hexdigest():
                        raise BuildError(
                            f"{name!r} already exists with different content — immutable artifacts are never overwritten"
                        ) from None

            writer = ChunkWriter(output.chunk_tokens, sink)
            with closing(_merged_part_stream(build, part_names, spool, merge_fanin)) as merged_stream:
                merged, _last = merge_parts([merged_stream], writer, max_tokens=output.token_budget)
            if output.token_budget is None:
                if merged != declared_docs or writer.total_tokens != declared_tokens:
                    raise BuildError(
                        f"output {output.name!r}: merged {merged} docs / {writer.total_tokens} tokens != "
                        f"markers' {declared_docs} / {declared_tokens} — reconciliation failed"
                    )
            elif writer.total_tokens > output.token_budget:
                raise BuildError(f"output {output.name!r}: token budget exceeded — exact-cut invariant broken")
            if not writer.chunks:
                if output.optional:
                    results[output.name] = {"skipped": "empty"}
                    continue
                raise BuildError(f"output {output.name!r}: no tokens — empty required output")

            manifest: dict[str, Any] = {
                "format": output.format,
                "dtype": "<i4" if output.format == "pithos_stream_v1" else "int32",
                "eos_id": recipe.eos_id,
                "chunk_tokens": output.chunk_tokens,
                "total_tokens": writer.total_tokens,
                "total_docs": merged,
                "chunks": writer.chunks,
            }
            manifest["manifest_sha256"] = manifest_sha256(manifest)
            Manifest(manifest)  # full structural/identity validation of what we are about to publish
            _write_immutable(build, f"{prefix}/manifest.json", _json_bytes(manifest))
            results[output.name] = {
                "docs": merged,
                "tokens": writer.total_tokens,
                "chunks": len(writer.chunks),
                "manifest_sha256": manifest["manifest_sha256"],
            }
        finalize = {"recipe_sha256": lock.recipe_sha256, "outputs": results}
        _write_immutable(build, "finalize.json", _json_bytes(finalize))
        return finalize
    finally:
        shutil.rmtree(spool, ignore_errors=True)


def _legacy_expected_tokens(lock: Lock) -> int:
    """The logical token count of the locked legacy inventory: non-final
    objects contribute exactly legacy_chunk_tokens (their +1 overlap tail is
    dropped); the final object contributes all of its tokens."""
    items = lock.source_lock["items"]
    c = int(lock.source_lock["legacy_chunk_tokens"])
    return sum(c if pos < len(items) - 1 else int(it["tokens"]) for pos, it in enumerate(items))


def _legacy_logical_stream(build: Storage, lock: Lock, spool: str) -> Generator[bytes, None, None]:
    """The exact flat token stream of a locked legacy binary corpus:
    non-final objects contribute their first legacy_chunk_tokens tokens
    (the +1 overlap tail is checked against the next object's head, then
    dropped); the final object contributes all of its tokens. Objects are
    processed ONE at a time through a verified descriptor (local files are
    hashed from the mmap'd descriptor itself) and memmap slices — bounded
    memory, and every spool file is reclaimed before the next object."""
    items = lock.source_lock["items"]
    c = int(lock.source_lock["legacy_chunk_tokens"])
    storage = storage_for(lock.source_lock["uri"])
    prev_tail: int | None = None
    for pos, raw_item in enumerate(items):
        item = SourceItem(raw_item["name"], int(raw_item["size"]), raw_item["sha256"])
        handle: ItemHandle | None = None
        try:
            if isinstance(storage, LocalStorage):
                handle = open_verified_local(storage.contained_path(item.name), item)
            else:
                with storage.open_read(item.name) as src:
                    path = spool_stream(src, item, spool)
                handle = ItemHandle(path=path, file=open(path, "rb"), spooled=path)
            arr = np.memmap(handle.file, dtype="<i4", mode="r")
            n = int(arr.size)
            if n != int(raw_item["tokens"]):
                raise BuildError(f"legacy object {item.name!r} holds {n} tokens, lock declared {raw_item['tokens']}")
            if pos < len(items) - 1:
                if n != c + 1:
                    raise BuildError(
                        f"legacy object {item.name!r} has {n} tokens, strict legacy geometry requires {c + 1}"
                    )
                logical_n = c
            else:
                if n > c + 1:
                    raise BuildError(f"final legacy object {item.name!r} has {n} tokens > {c + 1}")
                logical_n = n
            if prev_tail is not None and int(arr[0]) != prev_tail:
                raise BuildError(f"legacy stream discontinuity at {item.name!r}: overlap token mismatch")
            if pos < len(items) - 1:
                prev_tail = int(arr[c])
            for off in range(0, logical_n, LEGACY_SLICE_TOKENS):
                yield arr[off : min(off + LEGACY_SLICE_TOKENS, logical_n)].tobytes()
        finally:
            if handle is not None:
                handle.close()
                if handle.spooled is not None and os.path.exists(handle.spooled):
                    os.unlink(handle.spooled)


def _finalize_legacy(build: Storage, recipe: Recipe, lock: Lock) -> dict[str, Any]:
    """Byte-preserving re-chunk of a locked legacy flat-token stream into
    the approved output geometry. No re-tokenization, no reordering. The
    emitted logical total must reconcile with the locked ordered inventory
    (and with the recipe-pinned expected_logical_tokens when supplied) —
    the 'legacy' label is never asserted beyond the locked evidence."""
    expected_total = _legacy_expected_tokens(lock)
    pinned = lock.source_lock.get("expected_logical_tokens")
    if pinned is not None and expected_total != pinned:
        raise BuildError(
            f"locked legacy inventory holds {expected_total} logical tokens, "
            f"recipe pinned expected_logical_tokens={pinned} — identity mismatch"
        )
    spool = tempfile.mkdtemp(prefix="pithos-legacy-")
    try:
        results: dict[str, Any] = {}
        for output in _plan_outputs({"outputs": _output_payloads(recipe)}):
            prefix = f"outputs/{output.name}"

            def sink(_idx: int, blob: bytes, _p=prefix) -> None:
                name = f"{_p}/chunk-{_idx:05d}.bin"
                try:
                    build.write_bytes(name, blob, create_once=True)
                except ImmutableConflict:
                    if build.sha256(name) != hashlib.sha256(blob).hexdigest():
                        raise BuildError(f"{name!r} already exists with different content") from None

            writer = ChunkWriter(output.chunk_tokens, sink)
            budget = output.token_budget
            with closing(_legacy_logical_stream(build, lock, spool)) as slabs:
                for slab in slabs:
                    if budget is not None:
                        remaining = budget - writer.total_tokens
                        if remaining <= 0:
                            break
                        slab = slab[: 4 * min(len(slab) // 4, remaining)]
                    writer.write(slab)
            writer.close()
            if budget is not None and writer.total_tokens > budget:
                raise BuildError(f"output {output.name!r}: token budget exceeded — exact-cut invariant broken")
            if budget is None and writer.total_tokens != expected_total:
                raise BuildError(
                    f"output {output.name!r}: emitted {writer.total_tokens} logical tokens but the locked "
                    f"inventory holds {expected_total} — reconciliation failed"
                )
            if not writer.chunks:
                if output.optional:
                    results[output.name] = {"skipped": "empty"}
                    continue
                raise BuildError(f"output {output.name!r}: no tokens — empty required output")
            manifest = {
                "format": output.format,
                "dtype": "<i4" if output.format == "pithos_stream_v1" else "int32",
                "eos_id": recipe.eos_id,
                "chunk_tokens": output.chunk_tokens,
                "total_tokens": writer.total_tokens,
                "total_docs": 0,  # a re-chunk preserves tokens; document provenance lives in the source lock
                "chunks": writer.chunks,
            }
            manifest["manifest_sha256"] = manifest_sha256(manifest)
            Manifest(manifest)
            _write_immutable(build, f"{prefix}/manifest.json", _json_bytes(manifest))
            results[output.name] = {
                "docs": 0,
                "tokens": writer.total_tokens,
                "chunks": len(writer.chunks),
                "manifest_sha256": manifest["manifest_sha256"],
            }
        finalize = {"recipe_sha256": lock.recipe_sha256, "outputs": results}
        _write_immutable(build, "finalize.json", _json_bytes(finalize))
        return finalize
    finally:
        shutil.rmtree(spool, ignore_errors=True)


# ------------------------------------------------------------------ verify
def verify_build(build: Storage) -> dict[str, Any]:
    """Independent verification: re-validate every manifest, re-hash every
    chunk (streamed), re-check marker reconciliation, per-crawl
    declared-count reconciliation (declared counts come from the locked,
    ranged-scan row evidence — never from worker-derived counts), and
    train/validation split separation."""
    plan = _load_plan(build)
    recipe, lock = _plan_recipe_lock(plan)
    _enforce_code_tree(lock)
    if not build.exists("finalize.json"):
        raise BuildError("build is not finalized — run 'pithos build finalize' first")
    finalize = json.loads(build.read_bytes("finalize.json"))
    legacy = recipe.source["kind"] == "legacy_binary"
    roles = sorted({o["role"] for o in plan["outputs"]})
    all_markers = (
        []
        if legacy
        else [markers_mod.load_marker(build, u["unit_id"], lock, unit=u, roles=roles) for u in plan["units"]]
    )
    has_validation_output = any(o["role"] == "validation" for o in plan["outputs"])
    report: dict[str, Any] = {"outputs": {}}

    for o in plan["outputs"]:
        name = o["name"]
        if finalize["outputs"].get(name, {}).get("skipped"):
            report["outputs"][name] = {"skipped": "empty"}
            continue
        prefix = f"outputs/{name}"
        manifest = Manifest(json.loads(build.read_bytes(f"{prefix}/manifest.json")))
        if manifest.identity != finalize["outputs"][name]["manifest_sha256"]:
            raise BuildError(f"output {name!r}: manifest identity differs from finalize.json")
        for chunk in manifest.chunks:
            cname = f"{prefix}/{chunk.name}"
            size = 0
            h = hashlib.sha256()
            with build.open_read(cname) as f:
                while slab := f.read(1 << 20):
                    h.update(slab)
                    size += len(slab)
            if h.hexdigest() != chunk.sha256:
                raise BuildError(f"output {name!r}: chunk {chunk.name!r} sha256 mismatch — corrupt artifact")
            if size != 4 * chunk.tokens:
                raise BuildError(f"output {name!r}: chunk {chunk.name!r} size mismatch")
        if not legacy and has_validation_output:
            want_validation = o["role"] == "validation"
            for m in all_markers:
                with build.open_read(m["outputs"][o["role"]]["part"]) as part:
                    for key, _raw in read_records(part):
                        is_val = pol.is_validation(recipe.validation_domain, recipe.validation_fraction, key)
                        if is_val != want_validation:
                            raise BuildError(
                                f"output {name!r}: record {key.hex()} violates the validation reservation — split leak"
                            )
            declared_docs = sum(m["outputs"][o["role"]]["docs"] for m in all_markers)
            if o["token_budget"] is None and manifest.data["total_docs"] != declared_docs:
                raise BuildError(f"output {name!r}: manifest docs != markers' declared docs")
        report["outputs"][name] = {
            "manifest_sha256": manifest.identity,
            "chunks": len(manifest.chunks),
            "tokens": manifest.data["total_tokens"],
            "docs": manifest.data["total_docs"],
        }

    if not legacy:
        declared_by_crawl: dict[str, int] = {}
        seen_by_crawl: dict[str, int] = {}
        crawl_of_unit = {u["unit_id"]: u["crawl"] for u in plan["units"]}
        rows_of_unit = {u["unit_id"]: u.get("rows") for u in plan["units"]}
        for m in all_markers:
            crawl = crawl_of_unit[m["unit_id"]]
            rows = rows_of_unit[m["unit_id"]]
            if rows is not None:
                seen_by_crawl[crawl] = seen_by_crawl.get(crawl, 0) + m["stats"]["seen"]
                declared_by_crawl[crawl] = declared_by_crawl.get(crawl, 0) + rows
        for crawl, declared in sorted(declared_by_crawl.items()):
            seen = seen_by_crawl.get(crawl, 0)
            if seen != declared:  # stats["seen"] counts every parsed doc, EOS-dropped included
                raise BuildError(
                    f"crawl {crawl!r}: processed {seen} docs but the source declared {declared} rows — "
                    "per-crawl reconciliation failed"
                )
        report["crawls"] = {
            c: {"declared": d, "processed": seen_by_crawl.get(c, 0)} for c, d in sorted(declared_by_crawl.items())
        }
    _write_immutable(build, "verify.json", _json_bytes(report))
    return report


# ----------------------------------------------------------------- publish
def publish_build(build: Storage, *, client: Any = None, fetcher: Any = None) -> dict[str, Any]:
    """Immutably publish finalized, verified outputs to the lock's
    publish_root. Resumable: chunks already present at the destination are
    hash-verified and skipped (an interrupted publish retries cleanly), a
    DIFFERENT object under any name fails. The manifest is published LAST —
    its presence marks a complete publication, so an existing destination
    manifest is only honored after EVERY chunk it references is verified
    present and hash-correct (a dangling or corrupt publication fails
    loudly), and a concurrent identical publisher's manifest write is
    verified, never a failure."""
    plan = _load_plan(build)
    _recipe, lock = _plan_recipe_lock(plan)
    _enforce_code_tree(lock)
    if not lock.publish_root:
        raise BuildError("no publish destination pinned — the recipe's publish_root is null")
    if not build.exists("verify.json"):
        raise BuildError("build is not verified — run 'pithos build verify' first")
    finalize = json.loads(build.read_bytes("finalize.json"))
    dest = storage_for(lock.publish_root, client=client, fetcher=fetcher)
    published: dict[str, Any] = {}
    for o in plan["outputs"]:
        name = o["name"]
        info = finalize["outputs"].get(name, {})
        if info.get("skipped"):
            continue
        prefix = f"outputs/{name}"
        manifest_raw = build.read_bytes(f"{prefix}/manifest.json")
        manifest = Manifest(json.loads(manifest_raw))
        dest_manifest = f"{name}/manifest.json"
        entry = {"uri": f"{lock.publish_root}/{name}", "manifest_sha256": manifest.identity}
        if dest.exists(dest_manifest):
            existing = Manifest(json.loads(dest.read_bytes(dest_manifest)))
            if existing.identity != manifest.identity:
                raise BuildError(
                    f"publish destination already holds a DIFFERENT build of output {name!r} "
                    f"({existing.identity[:16]}… != {manifest.identity[:16]}…) — immutable, never overwritten"
                )
            for chunk in manifest.chunks:  # a manifest means COMPLETE — prove it before honoring it
                dest_name = f"{name}/{chunk.name}"
                if not dest.exists(dest_name):
                    raise BuildError(
                        f"publish destination holds the {name!r} manifest but chunk {chunk.name!r} is MISSING — "
                        "the publication is dangling, refusing to treat it as complete"
                    )
                if dest.sha256(dest_name) != chunk.sha256:
                    raise BuildError(
                        f"publish destination chunk {dest_name!r} fails the manifest sha256 under an existing "
                        "manifest — the publication is corrupt"
                    )
            published[name] = {**entry, "already_present": True}
            continue
        for chunk in manifest.chunks:
            dest_name = f"{name}/{chunk.name}"
            if dest.exists(dest_name):
                if dest.sha256(dest_name) != chunk.sha256:
                    raise BuildError(
                        f"publish destination chunk {dest_name!r} differs from the verified manifest — "
                        "refusing to publish over it"
                    )
                continue  # verified survivor of an interrupted publish
            try:
                with (
                    build.open_read(f"{prefix}/{chunk.name}") as src,
                    dest.open_write(dest_name, create_once=True) as dst,
                ):
                    while slab := src.read(1 << 20):
                        dst.write(slab)
            except ImmutableConflict:
                if dest.sha256(dest_name) != chunk.sha256:  # a concurrent publisher raced us
                    raise BuildError(f"publish destination chunk {dest_name!r} differs from the manifest") from None
        try:
            dest.write_bytes(dest_manifest, manifest_raw, create_once=True)
        except ImmutableConflict:  # a concurrent identical publisher wrote it first
            if dest.read_bytes(dest_manifest) != manifest_raw:
                raise BuildError(
                    f"publish destination manifest for {name!r} differs from this build — immutable, never overwritten"
                ) from None
        published[name] = {**entry, "already_present": False}
    # the persisted receipt is deterministic (no already_present flags), so
    # an identical re-publish is a no-op at the build root too
    receipt = {
        "publish_root": lock.publish_root,
        "outputs": {n: {k: v for k, v in e.items() if k != "already_present"} for n, e in published.items()},
    }
    _write_immutable(build, "publish.json", _json_bytes(receipt))
    return {"publish_root": lock.publish_root, "outputs": published}
