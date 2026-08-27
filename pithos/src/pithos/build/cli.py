"""The pithos build-plane CLI.

    pithos recipe lock RECIPE [--publish-root URI] [--source-uri URI]
    pithos build plan RECIPE [--build-root URI]
    pithos build worker BUILD_URI --worker-index N --worker-count M
    pithos build finalize BUILD_URI
    pithos build verify BUILD_URI
    pithos publish BUILD_URI

URIs are local paths (or file://), s3://bucket/prefix (with optional
?endpoint_url=...&profile=... for R2-style services — credentials come from
the environment's own config, never the URI), and http(s):// for immutable
source origins. The lock phase resolves the immutable pins (source revision
and per-object content digests, tokenizer asset digest and implementation
version, code revision and source-tree digest, per-corpus hash seed) and
fails clearly when a pin cannot be resolved — nothing is invented. Recipes
that leave publish_root or a legacy_binary source uri null MUST supply them
as explicit lock-time flags; a flag that disagrees with a recipe-pinned
value is an error. All phase errors are typed PithosErrors; the CLI never
raises SystemExit from library code.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys

from collections.abc import Sequence
from typing import Any

from ..errors import PithosError
from .codepin import pithos_tree_sha256
from .engine import finalize_build, plan_build, publish_build, run_worker, verify_build
from .recipe import load_lock, load_recipe, lock_path_for, lock_recipe, write_lock
from .sources import INVENTORY_SUFFIXES, HfHubClient, hf_locked_inventory
from .storage import storage_for
from .tokenize import ByteTokenizer, hf_tokenizer_evidence


def _resolve_source(recipe: Any, *, source_uri: str | None = None) -> dict[str, Any]:
    """Resolve the source's immutable evidence for the lock: per-object
    content sha256 for every selected item, plus the kind-specific identity
    (HF commit, legacy stream order)."""
    kind = recipe.source["kind"]
    if kind == "huggingface":
        client = HfHubClient()
        revision = recipe.source.get("revision")  # optional branch/tag pin in the recipe
        commit, items = hf_locked_inventory(
            client,
            recipe.source["dataset"],
            str(revision) if revision else "main",
            recipe.include_rules,
            recipe.exclude_rules,
        )
        return {"source_revision": commit, "source_items": items}
    if kind in {"local", "s3"}:
        storage = storage_for(recipe.source["uri"])
        items = []
        for o in storage.list(""):
            if not o.name.endswith(INVENTORY_SUFFIXES):
                continue
            items.append({"name": o.name, "size": o.size, "sha256": storage.sha256(o.name)})
        if not items:
            raise PithosError(f"source {recipe.source['uri']!r} has no .jsonl/.parquet items — nothing to lock")
        return {"source_items": items}
    if kind == "legacy_binary":
        recipe_uri = recipe.source.get("uri")
        if recipe_uri is not None and source_uri is not None and source_uri != recipe_uri:
            raise PithosError(f"--source-uri {source_uri!r} != the recipe's pinned source uri {recipe_uri!r}")
        uri = recipe_uri if isinstance(recipe_uri, str) and recipe_uri else source_uri
        if not isinstance(uri, str) or not uri:
            raise PithosError(
                "legacy_binary source uri is null — pass --source-uri URI with the locked legacy stream location"
            )
        storage = storage_for(uri)
        items = []
        for o in storage.list(""):
            if not o.name.endswith(".bin"):
                raise PithosError(f"legacy stream holds a non-binary object {o.name!r}")
            if o.size % 4:
                raise PithosError(f"legacy object {o.name!r} size is not whole int32 tokens")
            items.append({"name": o.name, "size": o.size, "sha256": storage.sha256(o.name), "tokens": o.size // 4})
        if not items:
            raise PithosError(f"legacy source {uri!r} is empty — nothing to lock")
        return {"source_items": items, "source_uri": uri}
    # http: fetch every declared item and pin its content digest AND byte count
    # (item names were already validated by the recipe's canonical name rule)
    storage = storage_for(recipe.source["base_url"])
    items = []
    for name in recipe.source["items"]:
        items.append({"name": name, **_hash_and_count(storage, name)})
    return {"source_items": items}


def _hash_and_count(storage: Any, name: str) -> dict[str, Any]:
    """Streaming sha256 + exact byte count of one immutable HTTP object —
    the lock pins real size evidence, never a placeholder 0."""
    h = hashlib.sha256()
    n = 0
    with storage.open_read(name) as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
            n += len(chunk)
    return {"size": n, "sha256": h.hexdigest()}


def _resolve_tokenizer_evidence(recipe: Any) -> dict[str, str]:
    if recipe.tokenizer_kind == "byte":
        return {
            "revision": ByteTokenizer.revision,
            "asset_sha256": ByteTokenizer.asset_sha256(),
            "impl_version": ByteTokenizer.impl_version,
        }
    return hf_tokenizer_evidence(recipe.tokenizer_name, recipe.tokenizer_rev, client=HfHubClient())


def _resolve_code_revision(explicit: str | None) -> str:
    if explicit:
        return explicit
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=30)
        return out.stdout.strip()
    except Exception as e:
        raise PithosError(f"cannot resolve the code revision via git ({e}) — pass --code-revision explicitly") from e


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pithos", description="Deterministic Pithos corpus builds")
    sub = parser.add_subparsers(dest="command", required=True)

    p_recipe = sub.add_parser("recipe", help="recipe phase")
    recipe_sub = p_recipe.add_subparsers(dest="recipe_command", required=True)
    p_lock = recipe_sub.add_parser("lock", help="resolve and pin the immutable build lock")
    p_lock.add_argument("recipe", metavar="RECIPE")
    p_lock.add_argument("--code-revision", default=None, help="explicit code revision (default: git HEAD)")
    p_lock.add_argument(
        "--publish-root",
        default=None,
        help="immutable publish destination URI — REQUIRED when the recipe's publish_root is null",
    )
    p_lock.add_argument(
        "--source-uri",
        default=None,
        help="legacy_binary stream location URI — REQUIRED when the recipe's source uri is null",
    )

    p_build = sub.add_parser("build", help="build phases")
    build_sub = p_build.add_subparsers(dest="build_command", required=True)
    p_plan = build_sub.add_parser("plan", help="select work units from the locked inventory, publish the plan")
    p_plan.add_argument("recipe", metavar="RECIPE")
    p_plan.add_argument("--build-root", default=None, help="build URI (default: ./builds/<recipe-name>)")
    p_worker = build_sub.add_parser("worker", help="process this worker's assigned units (resumable)")
    p_worker.add_argument("build_uri", metavar="BUILD_URI")
    p_worker.add_argument("--worker-index", type=int, required=True)
    p_worker.add_argument("--worker-count", type=int, required=True)
    for phase in ("finalize", "verify"):
        p = build_sub.add_parser(phase, help=f"{phase} the build")
        p.add_argument("build_uri", metavar="BUILD_URI")
    p_publish = sub.add_parser("publish", help="immutably publish a verified build")
    p_publish.add_argument("build_uri", metavar="BUILD_URI")

    args = parser.parse_args(argv)
    try:
        if args.command == "recipe":
            recipe = load_recipe(args.recipe)
            if args.source_uri is not None and recipe.source["kind"] != "legacy_binary":
                raise PithosError("--source-uri applies only to legacy_binary recipes")
            lock = lock_recipe(
                recipe,
                **_resolve_source(recipe, source_uri=args.source_uri),
                tokenizer_evidence=_resolve_tokenizer_evidence(recipe),
                code_revision=_resolve_code_revision(args.code_revision),
                code_tree_sha256=pithos_tree_sha256(),
                publish_root=args.publish_root,
            )
            written = write_lock(lock, lock_path_for(args.recipe))
            print(f"{'wrote' if written else 'already pinned'} {lock_path_for(args.recipe)}")
            return 0
        if args.command == "build":
            if args.build_command == "plan":
                recipe = load_recipe(args.recipe)
                lock = load_lock(lock_path_for(args.recipe))
                root = args.build_root or f"./builds/{recipe.name}"
                plan = plan_build(recipe, lock, storage_for(root))
                print(f"planned {len(plan['units'])} work units at {root}")
                return 0
            build = storage_for(args.build_uri)
            if args.build_command == "worker":
                stats = run_worker(build, args.worker_index, args.worker_count)
                print(
                    f"worker {args.worker_index}/{args.worker_count}: "
                    f"{len(stats['completed'])} completed, {len(stats['skipped'])} resumed"
                )
                return 0
            if args.build_command == "finalize":
                finalize_build(build)
                print("finalized")
                return 0
            verify_build(build)
            print("verified")
            return 0
        # publish
        receipt = publish_build(storage_for(args.build_uri))
        print(f"published {len(receipt['outputs'])} outputs to {receipt['publish_root']}")
        return 0
    except PithosError as e:
        print(f"pithos: error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
