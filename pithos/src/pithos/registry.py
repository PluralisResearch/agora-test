"""Named corpus registry: resolve a corpus NAME to its publication facts.

The registry is a YAML file mapping names to non-secret publication
configuration: the immutable manifest identity pin, optional reader cache
overrides, and optionally the corpus URI (local path, HTTP(S), or ``s3://``
— with optional ``endpoint_url``/``region``/``profile`` query parameters
for S3-compatible services such as R2). Entries may omit the URI so the
packaged registry never publishes the location; such corpora take their
location at run time (e.g. from the authorizer) and the entry pins what
that location must resolve to. The same corpus published in two places is
simply two entries.

Credentials never live in the registry: S3/R2 keys come from the
environment or the named AWS profile. Reader overrides exist because the
cache defaults fit 2**26-token chunks; corpora with 2**30-token (4 GiB)
chunks — legacy or pithos alike — need ``budget_bytes`` sized past one
chunk and prefetch disabled, or the prefetched neighbour and the current
chunk evict each other in a re-download loop.

YAML parsing uses the consumer's own PyYAML (training images have one) —
pithos deliberately does not depend on it, mirroring `pithos.torch`'s
contract with torch.
"""

from __future__ import annotations

import os

from dataclasses import dataclass
from typing import Any

from .errors import RegistryError


DEFAULT_REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpora.yaml")

_ENTRY_KEYS = {"uri", "manifest_identity", "reader", "description"}
_READER_KEYS = {"budget_bytes", "prefetch_depth"}


@dataclass(frozen=True)
class CorpusEntry:
    """One validated registry entry: how to hold a published corpus to its
    identity, and optionally where it lives. ``uri`` is None for corpora
    whose location is supplied at run time (e.g. by the authorizer) so the
    packaged registry never publishes the URL."""

    name: str
    uri: str | None
    manifest_identity: str | None
    budget_bytes: int | None
    prefetch_depth: int | None
    description: str


def _positive_int(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RegistryError(f"registry {label} must be an int >= {minimum}, got {value!r}")
    return value


def _entry_from_raw(name: str, raw: Any) -> CorpusEntry:
    if not isinstance(raw, dict):
        raise RegistryError(f"registry entry {name!r} must be a mapping")
    unknown = set(raw) - _ENTRY_KEYS
    if unknown:
        raise RegistryError(f"registry entry {name!r} has unknown fields {sorted(map(repr, unknown))}")
    uri = raw.get("uri")
    if uri is not None and (not isinstance(uri, str) or not uri):
        raise RegistryError(f"registry entry {name!r}: uri must be a non-empty string when present")
    identity = raw.get("manifest_identity")
    if identity is not None:
        if not isinstance(identity, str) or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
            raise RegistryError(f"registry entry {name!r}: manifest_identity must be 64 lowercase hex characters")
    elif uri is None:
        # Without a URI the entry exists solely to pin the corpus a run-time
        # location must resolve to; identity-less it would pin nothing.
        raise RegistryError(f"registry entry {name!r}: entries without a uri must pin manifest_identity")
    elif uri.split("://", 1)[0].lower() in {"http", "https", "s3"}:
        # A remote object can be replaced under the same URI; a named remote
        # corpus without its identity pin would silently follow the swap.
        raise RegistryError(f"registry entry {name!r}: remote corpora must pin manifest_identity")
    reader = raw.get("reader")
    if reader is None:
        reader = {}
    if not isinstance(reader, dict):
        raise RegistryError(f"registry entry {name!r}: reader must be a mapping")
    unknown = set(reader) - _READER_KEYS
    if unknown:
        raise RegistryError(f"registry entry {name!r}: reader has unknown fields {sorted(map(repr, unknown))}")
    budget = reader.get("budget_bytes")
    prefetch = reader.get("prefetch_depth")
    return CorpusEntry(
        name=name,
        uri=uri,
        manifest_identity=identity,
        budget_bytes=None if budget is None else _positive_int(budget, f"{name!r} reader.budget_bytes", 1),
        prefetch_depth=None if prefetch is None else _positive_int(prefetch, f"{name!r} reader.prefetch_depth", 0),
        description=str(raw.get("description", "")),
    )


def load_registry(path: str | None = None) -> dict[str, CorpusEntry]:
    """All entries of a registry file, fully validated.

    Raises:
        RegistryError: if PyYAML is unavailable, the file is unreadable, or
            any entry fails validation.
    """
    registry_path = DEFAULT_REGISTRY_PATH if path is None else path
    try:
        import yaml  # noqa: PLC0415 — the consumer's own PyYAML, never a pithos dependency
    except ModuleNotFoundError as e:
        raise RegistryError(
            "the corpus registry needs PyYAML, which pithos never installs: pip install pyyaml in your own environment"
        ) from e
    try:
        with open(registry_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except OSError as e:
        raise RegistryError(f"cannot read corpus registry {registry_path!r}: {e.strerror or type(e).__name__}") from e
    except (yaml.YAMLError, UnicodeDecodeError, ValueError) as e:
        raise RegistryError(f"corpus registry {registry_path!r} is not valid YAML: {type(e).__name__}") from e
    if not isinstance(data, dict) or not isinstance(data.get("corpora"), dict) or not data["corpora"]:
        raise RegistryError(f"corpus registry {registry_path!r} must hold a non-empty 'corpora' mapping")
    for name in data["corpora"]:
        # Non-string names invite silent collisions (YAML 1 vs "1" both
        # coerce to "1"); YAML itself already keeps only the last duplicate
        # of identical keys, which no loader can recover after the fact.
        if not isinstance(name, str):
            raise RegistryError(f"registry corpus names must be strings, got {name!r}")
    return {name: _entry_from_raw(name, raw) for name, raw in data["corpora"].items()}


def resolve_corpus(name: str, path: str | None = None) -> CorpusEntry:
    """The validated entry for `name`.

    Raises:
        RegistryError: if the registry is invalid or the name is unknown.
    """
    entries = load_registry(path)
    entry = entries.get(name)
    if entry is None:
        raise RegistryError(f"unknown corpus {name!r}; registry holds {sorted(entries)}")
    return entry
