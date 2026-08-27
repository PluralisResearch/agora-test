"""Versioned, recipe-controlled document transforms.

A recipe names transforms as {"name", "version"} pairs; the lock pins their
identity (name@version plus a digest of the spec) so a build always runs the
exact transform set it was locked with. The registry is deliberately small
and explicit — unknown names/versions are rejected at recipe-load time, and
FineWeb-Edu pins an EMPTY transform list (upstream filtering and per-crawl
MinHash dedup are preserved as-is; no global dedup, no legacy repetition
filter — those are not transforms and are never implemented here).
"""

from __future__ import annotations

import hashlib
import json
import re

from collections.abc import Callable
from dataclasses import dataclass

from ..errors import BuildError


@dataclass(frozen=True)
class TransformSpec:
    name: str
    version: int

    @property
    def identity(self) -> str:
        return f"{self.name}@{self.version}"

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "version": self.version}


def _whitespace_normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


_REGISTRY: dict[str, Callable[[str], str]] = {
    "whitespace_normalize@1": _whitespace_normalize,
}


def transform_identity_digest(specs: tuple[TransformSpec, ...]) -> str:
    """Canonical digest of a transform list — pinned into the lock."""
    canon = json.dumps([s.to_dict() for s in specs], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canon).hexdigest()


def resolve_transforms(raw: object) -> tuple[TransformSpec, ...]:
    """Validate a recipe's transform list against the registry."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise BuildError("transforms must be a list of {name, version} objects")
    out: list[TransformSpec] = []
    for entry in raw:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("name"), str)
            or not isinstance(entry.get("version"), int)
        ):
            raise BuildError(f"malformed transform spec {entry!r} — expected {{'name': str, 'version': int}}")
        spec = TransformSpec(entry["name"], entry["version"])
        if spec.identity not in _REGISTRY:
            raise BuildError(f"unknown transform {spec.identity!r} — registered: {sorted(_REGISTRY)}")
        out.append(spec)
    return tuple(out)


def apply_transforms(specs: tuple[TransformSpec, ...], text: str) -> str:
    """Apply the locked transform chain, in recipe order."""
    for spec in specs:
        text = _REGISTRY[spec.identity](text)
    return text
