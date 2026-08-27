"""Part completion markers: the resume/integrity contract of a build.

Every completed work unit publishes a marker pinning the recipe lock, source
lock, tokenizer lock, code revision AND source-tree digest, the plan unit's
EXACT source-item identity (name, size, sha256 — not merely a name), per-
output document/token/count stats, and each part artifact's sha256 under the
unit's own canonical part names. Worker index/count are NOT pinned:
assignment is ephemeral, and markers must be byte-identical across worker
counts and retry reassignments of the same plan.

A marker is create-once: re-running a unit with identical content is a
no-op (safe resume); a DIFFERENT marker for the same unit, a pin mismatch
against the build's lock or plan unit, a count invariant violation, or a
part whose bytes fail the pinned checksum is rejected loudly — never
silently rebuilt over.
"""

from __future__ import annotations

import json

from typing import Any

from ..errors import BuildError
from .recipe import Lock
from .storage import ImmutableConflict, Storage


MARKER_VERSION = 3


def marker_name(unit_id: str) -> str:
    if "/" in unit_id or unit_id.startswith("."):
        raise BuildError(f"unsafe unit id {unit_id!r}")
    return f"markers/{unit_id}.json"


def part_name(unit_id: str, role: str) -> str:
    """The ONE canonical part object name for a unit/role — markers bind to
    exactly this, never to another unit's checksum-valid part."""
    marker_name(unit_id)  # same unit-id safety rule
    if not role or "/" in role:
        raise BuildError(f"unsafe output role {role!r}")
    return f"parts/{unit_id}.{role}.partbin"


def build_marker(
    *,
    lock: Lock,
    unit_id: str,
    item: dict[str, Any],
    stats: dict[str, int],
    outputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Assemble a marker payload. `item` is the plan unit's exact source-
    item payload {"name", "size", "sha256"}; `stats` are the unit-level
    counts (seen/selected/dropped_eos/validation/tokens); `outputs` maps
    output role to {docs, tokens, part, part_sha256}."""
    return {
        "marker_version": MARKER_VERSION,
        "unit_id": unit_id,
        "item": item,
        "recipe_sha256": lock.recipe_sha256,
        "source_sha256": lock.source_sha256,
        "tokenizer_sha256": lock.tokenizer_sha256,
        "code_revision": lock.code_revision,
        "code_tree_sha256": lock.code_tree_sha256,
        "stats": stats,
        "outputs": outputs,
    }


def marker_bytes(marker: dict[str, Any]) -> bytes:
    return (json.dumps(marker, indent=2, sort_keys=True) + "\n").encode()


def write_marker(storage: Storage, marker: dict[str, Any]) -> bool:
    """Atomic create-once marker publication. Returns True if written, False
    if an identical marker already existed (idempotent resume — including
    the race where another worker published it first). Raises BuildError if
    a DIFFERENT marker exists for the unit."""
    data = marker_bytes(marker)
    name = marker_name(marker["unit_id"])
    try:
        storage.write_bytes(name, data, create_once=True)
        return True
    except ImmutableConflict:
        if storage.read_bytes(name) == data:
            return False
        raise BuildError(f"marker for unit {marker['unit_id']!r} already exists with different content") from None


def load_marker(
    storage: Storage, unit_id: str, lock: Lock, *, unit: dict[str, Any], roles: list[str]
) -> dict[str, Any]:
    """Read and fully validate a unit's marker. Every lock pin must match;
    the marker must be bound to the plan unit's EXACT source-item identity
    (name/size/sha256 — a stale marker naming another item is rejected even
    when its parts are checksum-valid); the outputs must cover exactly the
    plan's roles under the unit's own canonical part names; every pinned
    part checksum must verify (streamed — parts are never read whole); and
    the count invariants derivable without re-reading token bodies must
    hold. Raises BuildError on any mismatch or corruption."""
    name = marker_name(unit_id)
    if not storage.exists(name):
        raise BuildError(f"no completion marker for unit {unit_id!r}")
    try:
        marker = json.loads(storage.read_bytes(name))
    except json.JSONDecodeError as e:
        raise BuildError(f"marker for unit {unit_id!r} is corrupt: {e}") from e

    def bad(msg: str) -> BuildError:
        return BuildError(f"marker for unit {unit_id!r}: {msg}")

    if marker.get("marker_version") != MARKER_VERSION:
        raise bad("unsupported marker_version")
    if marker.get("unit_id") != unit_id:
        raise bad(f"unit id mismatch {marker.get('unit_id')!r}")
    if marker.get("recipe_sha256") != lock.recipe_sha256:
        raise bad("recipe lock mismatch — the build was planned under a different recipe")
    if marker.get("source_sha256") != lock.source_sha256:
        raise bad("source lock mismatch")
    if marker.get("tokenizer_sha256") != lock.tokenizer_sha256:
        raise bad("tokenizer lock mismatch")
    if marker.get("code_revision") != lock.code_revision:
        raise bad("code revision mismatch")
    if marker.get("code_tree_sha256") != lock.code_tree_sha256:
        raise bad("code source-tree digest mismatch")

    plan_item = unit["item"]
    item = marker.get("item")
    if not isinstance(item, dict):
        raise bad("source item identity missing")
    for field in ("name", "size", "sha256"):
        if item.get(field) != plan_item[field]:
            raise bad(
                f"item {field} is {item.get(field)!r}, plan unit pins {plan_item[field]!r} — "
                "the marker is not bound to this unit's exact source item"
            )

    stats = marker.get("stats")
    if not isinstance(stats, dict):
        raise bad("stats missing")
    for field in ("seen", "selected", "dropped_eos", "validation", "tokens"):
        if not isinstance(stats.get(field), int) or isinstance(stats[field], bool) or stats[field] < 0:
            raise bad(f"stats.{field} must be a non-negative int")
    outputs = marker.get("outputs")
    if not isinstance(outputs, dict):
        raise bad("outputs missing")
    if set(outputs) != set(roles):
        raise bad(f"output roles {sorted(outputs)} != the plan's {sorted(roles)}")
    for oname, o in outputs.items():
        if not isinstance(o, dict) or not isinstance(o.get("part"), str):
            raise bad(f"output {oname!r}: part missing")
        if o["part"] != part_name(unit_id, oname):
            raise bad(f"output {oname!r}: part {o['part']!r} is not this unit's own {part_name(unit_id, oname)!r}")
        for field in ("docs", "tokens"):
            if not isinstance(o.get(field), int) or isinstance(o[field], bool) or o[field] < 0:
                raise bad(f"output {oname!r}: {field} must be a non-negative int")
        sha = o.get("part_sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise bad(f"output {oname!r}: bad part_sha256")
        if not storage.exists(o["part"]):
            raise bad(f"output {oname!r}: part {o['part']!r} is missing")
        if storage.sha256(o["part"]) != sha:
            raise bad(
                f"output {oname!r}: part {o['part']!r} checksum mismatch — corrupt artifact, refusing to resume over it"
            )

    if stats["seen"] != stats["selected"] + stats["dropped_eos"]:
        raise bad("count invariant broken: seen != selected + dropped_eos")
    if stats["validation"] > stats["selected"]:
        raise bad("count invariant broken: validation > selected")
    if sum(o["docs"] for o in outputs.values()) != stats["selected"]:
        raise bad("count invariant broken: role docs do not sum to stats.selected")
    if sum(o["tokens"] for o in outputs.values()) != stats["tokens"]:
        raise bad("count invariant broken: role tokens do not sum to stats.tokens")
    if "validation" in outputs and outputs["validation"]["docs"] != stats["validation"]:
        raise bad("count invariant broken: validation role docs != stats.validation")
    for oname, o in outputs.items():
        if o["tokens"] < 2 * o["docs"]:
            raise bad(f"output {oname!r}: tokens < 2*docs — below the [BOS]+[EOS] floor of a selected doc")
    rows = unit.get("rows")
    if rows is not None and stats["seen"] != rows:
        raise bad(f"stats.seen {stats['seen']} != the unit's locked {rows} source rows")
    return marker
