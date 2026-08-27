"""Pithos exception hierarchy.

All validation failures raise a subclass of `PithosError`, which itself
subclasses `ValueError` so existing ValueError handling keeps working.
Library code never raises bare `assert` or `SystemExit`.
"""

from __future__ import annotations


class PithosError(ValueError):
    """Base class for all pithos validation and integrity errors."""


class CorpusError(PithosError):
    """A deterministic corpus-build primitive was violated: bad record
    framing, unsorted or duplicate keys, out-of-range token ids, invalid
    arguments, or a malformed part buffer."""


class ManifestError(PithosError):
    """A manifest failed structural, identity, or chunk-entry validation:
    unknown format, wrong dtype, unsafe or duplicate chunk names, invalid
    sizes, missing/malformed sha256, a mismatched identity hash, or an
    unsupported sequence length."""


class CacheError(PithosError):
    """A chunk object failed integrity verification (size or sha256) or a
    download could not be published atomically."""


class DownloadError(CacheError):
    """A chunk object transfer failed before publication: an unsupported or
    malformed URL, an HTTP/S3 error, a stale or ignored range, or a
    truncated stream. Carries `resumable`: when True, the verified partial
    prefix is retained and the next attempt continues from it."""

    def __init__(self, message: str, *, resumable: bool) -> None:
        super().__init__(message)
        self.resumable = resumable


class RegistryError(PithosError):
    """A named-corpus registry entry is missing or malformed: unknown name,
    unreadable or invalid YAML, a non-string URI, a malformed identity pin,
    or an unrecognized entry field."""


class BuildError(PithosError):
    """A build-plane phase failed: an unresolvable or mismatched lock, an
    unsafe or corrupted part/marker, a reconciliation mismatch, or an
    attempt to overwrite a different immutable artifact."""
