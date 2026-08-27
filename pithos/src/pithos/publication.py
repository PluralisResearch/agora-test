"""Resolve an immutable published corpus into a local manifest and chunk locators."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import urllib.parse
import urllib.request

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from . import transport
from .errors import DownloadError, ManifestError
from .manifest import Manifest


_MAX_MANIFEST_BYTES = 64 << 20
_MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class ResolvedPublication:
    """A verified local manifest plus exact locators for all of its chunks."""

    manifest_path: str
    chunk_locators: dict[str, str]


def _read_limited(stream: BinaryIO, label: str) -> bytes:
    data = bytearray()
    while len(data) <= _MAX_MANIFEST_BYTES:
        block = stream.read(min(1 << 20, _MAX_MANIFEST_BYTES + 1 - len(data)))
        if not block:
            return bytes(data)
        data.extend(block)
    raise ManifestError(f"{label} exceeds {_MAX_MANIFEST_BYTES} bytes")


def _manifest_from_bytes(data: bytes, label: str) -> Manifest:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"{label} is not valid UTF-8 JSON") from exc
    return Manifest(value)


def _load_cached(path: str, expected_identity: str) -> Manifest | None:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        raise ManifestError("cached manifest is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ManifestError(f"cannot safely open cached manifest: {type(exc).__name__}") from exc
    with os.fdopen(fd, "rb") as stream:
        manifest = _manifest_from_bytes(_read_limited(stream, "cached manifest"), "cached manifest")
    if manifest.identity != expected_identity:
        raise ManifestError(
            f"cached manifest identity {manifest.identity[:16]}… != expected {expected_identity[:16]}…"
        )
    return manifest


def _cache_manifest(data: bytes, manifest: Manifest, cache_dir: str) -> str:
    root = os.path.realpath(cache_dir)
    manifest_dir = os.path.join(root, ".manifests")
    if os.path.islink(manifest_dir):
        raise ManifestError("manifest cache directory is a symlink")
    try:
        os.makedirs(manifest_dir, exist_ok=True)
    except OSError as exc:
        raise ManifestError(f"cannot create manifest cache: {type(exc).__name__}") from exc
    if os.path.realpath(manifest_dir) != os.path.join(root, ".manifests"):
        raise ManifestError("manifest cache directory escapes cache_dir")

    path = os.path.join(manifest_dir, f"{manifest.identity}.json")
    existing = _load_cached(path, manifest.identity)
    if existing is not None:
        return path

    fd, temporary = tempfile.mkstemp(prefix=".manifest-", dir=manifest_dir)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if _load_cached(path, manifest.identity) is None:
                raise ManifestError("manifest publication race produced no cached manifest") from None
        return path
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _remote_root(corpus_uri: str) -> tuple[urllib.parse.SplitResult, str, str]:
    try:
        parts = urllib.parse.urlsplit(corpus_uri)
    except ValueError as exc:
        raise ManifestError("malformed corpus URI") from exc
    if parts.fragment:
        raise ManifestError("corpus URI must not contain a fragment")
    if parts.query and parts.scheme.lower() != "s3":
        raise ManifestError("only s3 corpus URIs may carry query parameters")
    path = parts.path.rstrip("/")
    if path.endswith(f"/{_MANIFEST_NAME}") or path == _MANIFEST_NAME:
        root_path = path[: -len(_MANIFEST_NAME)].rstrip("/")
    else:
        root_path = path
    manifest_path = f"{root_path}/{_MANIFEST_NAME}" if root_path else f"/{_MANIFEST_NAME}"
    return parts, root_path, manifest_path


def _local_publication(
    corpus_uri: str,
    cache_dir: str,
    expected_manifest_identity: str | None,
) -> ResolvedPublication:
    if urllib.parse.urlsplit(corpus_uri).scheme.lower() == "file":
        parts = urllib.parse.urlsplit(corpus_uri)
        if parts.netloc not in {"", "localhost"} or parts.query or parts.fragment:
            raise ManifestError("file corpus URI must be local and must not contain a query or fragment")
        value = urllib.request.url2pathname(parts.path)
    else:
        value = corpus_uri
    candidate = Path(value)
    manifest_path = candidate if candidate.name == _MANIFEST_NAME else candidate / _MANIFEST_NAME
    try:
        with open(manifest_path, "rb") as stream:
            manifest = _manifest_from_bytes(_read_limited(stream, "local manifest"), "local manifest")
    except ManifestError:
        raise
    except OSError as exc:
        raise ManifestError(f"cannot read local manifest: {type(exc).__name__}") from exc
    if expected_manifest_identity is not None and manifest.identity != expected_manifest_identity:
        raise ManifestError(
            f"manifest identity {manifest.identity[:16]}… != expected {expected_manifest_identity[:16]}…"
        )
    root = manifest_path.parent.resolve()
    if root == Path(cache_dir).resolve():
        locators: dict[str, str] = {}
    else:
        locators = {name: str(root / name) for name in manifest.chunk_names}
    return ResolvedPublication(str(manifest_path), locators)


def resolve_publication(
    corpus_uri: str,
    cache_dir: str,
    expected_manifest_identity: str | None = None,
) -> ResolvedPublication:
    """Resolve a local, HTTP(S), or S3 publication root or manifest URI.

    Args:
        corpus_uri: Publication root URI, or its ``manifest.json`` URI.
        cache_dir: Runtime cache directory; remote manifests are cached beneath it.
        expected_manifest_identity: Optional immutable manifest identity pin.

    Returns:
        A verified local manifest path and a complete chunk-locator mapping.
    """
    if not isinstance(corpus_uri, str) or not corpus_uri:
        raise ManifestError("corpus_uri must be a non-empty string")
    if expected_manifest_identity is not None:
        if (
            not isinstance(expected_manifest_identity, str)
            or len(expected_manifest_identity) != 64
            or any(c not in "0123456789abcdef" for c in expected_manifest_identity)
        ):
            raise ManifestError("expected_manifest_identity must be 64 lowercase hexadecimal characters")

    try:
        scheme = urllib.parse.urlsplit(corpus_uri).scheme.lower()
    except ValueError as exc:
        raise ManifestError("malformed corpus URI") from exc
    if scheme not in {"", "file", "http", "https", "s3"}:
        raise ManifestError(f"unsupported corpus URI scheme {scheme!r}")
    if scheme in {"", "file"}:
        return _local_publication(corpus_uri, cache_dir, expected_manifest_identity)

    parts, root_path, manifest_path = _remote_root(corpus_uri)
    if parts.scheme in {"http", "https"}:
        if not parts.netloc or parts.username is not None or parts.password is not None:
            raise ManifestError("HTTP corpus URI must have a host and must not contain credentials")
        root_uri = urllib.parse.urlunsplit((parts.scheme, parts.netloc, root_path, "", ""))
        manifest_uri = urllib.parse.urlunsplit((parts.scheme, parts.netloc, manifest_path, "", ""))
        locator = lambda name: f"{root_uri}/{urllib.parse.quote(name, safe='')}"
        manifest_locator = manifest_uri
    else:
        bucket = parts.netloc
        if not bucket or "@" in bucket:
            raise ManifestError("S3 corpus URI must contain a bucket and must not contain credentials")
        try:
            transport.s3_locator_params(parts.query)  # fail fast, before anything is fetched
        except DownloadError as exc:
            raise ManifestError(str(exc)) from None
        prefix = root_path.lstrip("/")
        manifest_key = manifest_path.lstrip("/")
        suffix = f"?{parts.query}" if parts.query else ""
        locator = lambda name: f"s3://{bucket}/{prefix + '/' if prefix else ''}{name}{suffix}"
        manifest_locator = f"s3://{bucket}/{manifest_key}{suffix}"

    cached_path = None
    if expected_manifest_identity is not None:
        cached_path = os.path.join(os.path.realpath(cache_dir), ".manifests", f"{expected_manifest_identity}.json")
        cached = _load_cached(cached_path, expected_manifest_identity)
        if cached is not None:
            return ResolvedPublication(cached_path, {name: locator(name) for name in cached.chunk_names})

    raw = transport.read_small(manifest_locator, max_bytes=_MAX_MANIFEST_BYTES)
    manifest = _manifest_from_bytes(raw, "remote manifest")
    if expected_manifest_identity is not None and manifest.identity != expected_manifest_identity:
        raise ManifestError(
            f"manifest identity {manifest.identity[:16]}… != expected {expected_manifest_identity[:16]}…"
        )
    cached_path = _cache_manifest(raw, manifest, cache_dir)
    return ResolvedPublication(cached_path, {name: locator(name) for name in manifest.chunk_names})
