"""Byte transport for chunk objects: local paths, file/HTTP(S) URLs, and
lazily-imported S3 — with checksum-safe resumable partial downloads.

The caller (ChunkCache, under the per-chunk interprocess lock) hands over an
already-claimed temp descriptor plus `have` validated prefix bytes whose
sha256 is `prefix_sha256`. `fetch` completes the object:

- HTTP/file URLs go through urllib with an explicit finite timeout (tests
  patch it at exactly that seam). A
  resume sends `Range: bytes=have-` plus `If-Range` from the sidecar's
  strong ETag (else Last-Modified). A 206 with an aligned Content-Range is
  appended; a 200 (or a file:// response, which carries no status) means the
  range was ignored and the body IS the full object, streamed from byte
  zero; a 412/416 (`_RangeStaleError`) means the object changed under the partial
  and triggers one plain re-GET.
- Plain local paths resume by seek, using (ino, size, mtime_ns) as the
  change stamp — the If-Range equivalent for files.
- s3://bucket/key uses a lazily imported boto3 (`Range` + `IfMatch`);
  botocore-style 412/416/InvalidRange codes map to the same stale path.

Progress is journaled to a sidecar `{bytes, sha256-of-prefix, source_id,
etag, last_modified}` written AFTER the bytes it vouches for are on disk,
so a crash can only leave the file LONGER than the sidecar claims — the
next attempt truncates to the vouched prefix, never trusts unvouched bytes.
The journal keys on the manifest-object identity, never a locator, so a
refreshed signed URL or a mirror change still resumes. The final size +
sha256 check against the manifest stays with the cache, which is also the
only code that publishes.

NumPy-free; raises typed `DownloadError` (a CacheError) with a `resumable`
flag telling the cache whether to keep the partial for the next attempt.
Error text is locator-free: a scheme-plus-identity label stands in for the
raw URL, query string, credentials, S3 key, or local path.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import SplitResult, parse_qs, urlsplit

from .errors import DownloadError


_BLOCK = 1 << 20  # stream/granularity of the resume journal: 1 MiB
_URL_TIMEOUT = 30.0  # seconds; explicit and finite, so a stalled source cannot hang a worker
_USER_AGENT = "pithos"  # required by r2.dev
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CRANGE_RE = re.compile(r"bytes (\d+)-(\d+)/(?:\d+|\*)")


# ---------------------------------------------------------------- sidecar
def object_identity(expected_bytes: int, expected_sha256: str) -> str:
    """The stable, non-secret identity of one manifest object: a
    domain-separated SHA-256 over its expected byte count and content hash,
    as 64 lowercase hex. Locator-independent, so a signed-URL refresh or an
    R2/S3 mirror change for the same object keeps its resume identity."""
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes < 1:
        raise ValueError(f"expected_bytes must be a positive int, got {expected_bytes!r}")
    if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(expected_sha256):
        raise ValueError(f"expected_sha256 must be 64 lowercase hex, got {expected_sha256!r}")
    material = f"pithos-object-identity-v1:{expected_bytes}:{expected_sha256}"
    return hashlib.sha256(material.encode("ascii")).hexdigest()


def load_state(path: str) -> dict[str, Any] | None:
    """A validated resume sidecar, or None on ANY deviation (absent, torn,
    wrong types, planted symlink, legacy raw-url schema) — callers treat
    None as 'start over'."""
    try:
        if os.path.islink(path):
            return None
        with open(path) as f:
            state = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(state, dict) or "url" in state:
        return None  # the legacy raw-locator schema is never adopted
    nbytes = state.get("bytes")
    sha = state.get("sha256")
    source_id = state.get("source_id")
    if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0:
        return None
    if not isinstance(sha, str) or not _SHA256_RE.fullmatch(sha):
        return None
    if not isinstance(source_id, str) or not _SHA256_RE.fullmatch(source_id):
        return None
    for opt in ("etag", "last_modified"):
        if state.get(opt) is not None and not isinstance(state.get(opt), str):
            return None
    return {
        "bytes": nbytes,
        "sha256": sha,
        "source_id": source_id,
        "etag": state.get("etag"),
        "last_modified": state.get("last_modified"),
    }


def save_state(path: str, state: dict[str, Any]) -> None:
    """Journal the sidecar atomically (tmp + rename), never following a
    planted symlink at the journal temp path."""
    tmp = f"{path}.{os.getpid()}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)  # replaces (never follows) anything planted at path


def delete_quiet(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


# ---------------------------------------------------------------- opening
class _RangeStaleError(Exception):
    """The server rejected our range/precondition (412/416/InvalidRange):
    the object changed under the partial — restart from byte zero."""


@dataclass
class _Opened:
    """A scheme-neutral open object body: a readable stream plus the
    metadata the resume logic needs. `range_start` is the parsed
    Content-Range start, None when the header is absent."""

    stream: Any
    status: int | None
    etag: str | None
    last_modified: str | None
    range_start: int | None
    close: Callable[[], None]


def _header(resp: Any, name: str) -> str | None:
    headers = getattr(resp, "headers", None)
    if headers is None:
        return None
    value = headers.get(name)
    return str(value) if value is not None else None


def _content_range_start(value: str | None) -> int | None:
    if value is None:
        return None
    m = _CRANGE_RE.fullmatch(value.strip())
    return int(m.group(1)) if m else None


def _if_range_value(state: dict[str, Any] | None) -> str | None:
    """A strong validator for If-Range: the ETag unless it is weak, else
    Last-Modified. Without either, the range is sent bare and the final
    manifest sha256 remains the authority."""
    if not state:
        return None
    etag = state.get("etag")
    if etag and not etag.startswith("W/"):
        return etag
    return state.get("last_modified")


def _split_locator(url: str) -> SplitResult:
    """urlsplit that never surfaces a raw ValueError: malformed locator
    syntax (an unterminated IPv6 bracket, an NFKC-invalid netloc) becomes a
    non-resumable DownloadError. Context is suppressed because the ValueError
    text can itself embed the netloc."""
    try:
        return urlsplit(url)
    except ValueError:
        raise DownloadError("malformed chunk locator syntax", resumable=False) from None


def _safe_label(url: str, source_id: str | None) -> str:
    """A non-secret handle for error text: the normalized scheme plus a
    short prefix of the manifest-object identity when known — never the
    locator, query string, credentials, or local path."""
    try:
        scheme = urlsplit(url).scheme.lower()
    except ValueError:
        scheme = ""
    kind = scheme if scheme not in ("", "file") else "local"
    label = f"{kind} source"
    return f"{label} {source_id[:12]}" if source_id else label


def _urllib_open(url: str, have: int, state: dict[str, Any] | None, source_id: str) -> _Opened:
    headers: dict[str, str] = {"User-Agent": _USER_AGENT}
    if have > 0:
        headers["Range"] = f"bytes={have}-"
        if_range = _if_range_value(state)
        if if_range:
            headers["If-Range"] = if_range
    label = _safe_label(url, source_id)
    # one positional arg plus an explicit finite timeout: tests patch urlopen
    # with fakes accepting both
    try:
        resp = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=_URL_TIMEOUT)
    except urllib.error.HTTPError as e:
        if e.code in (412, 416):
            raise _RangeStaleError() from e
        # the reason phrase is server-controlled text and can echo the locator
        raise DownloadError(f"GET {label} failed: HTTP {e.code}", resumable=have > 0) from e
    except Exception as e:
        # the exception text may embed the full locator — forward its type only
        raise DownloadError(f"GET {label} failed: {type(e).__name__}", resumable=have > 0) from e
    if hasattr(resp, "__enter__"):  # real addinfourl and CM fakes alike

        def close(resp: Any = resp) -> None:
            resp.__exit__(None, None, None)

        stream = resp.__enter__()
    else:
        stream, close = resp, resp.close
    return _Opened(
        stream=stream,
        status=getattr(resp, "status", None),
        etag=_header(resp, "ETag"),
        last_modified=_header(resp, "Last-Modified"),
        range_start=_content_range_start(_header(resp, "Content-Range")),
        close=close,
    )


def s3_locator_params(query: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validated (Session kwargs, client kwargs) from an s3 locator query.

    The supported parameters mirror the build plane's storage convention and
    carry non-secret endpoint configuration only: ``endpoint_url`` (an
    S3-compatible service such as R2), ``region``, and ``profile``.
    Credentials never travel in locators; anything unrecognized — especially
    credential-shaped keys — is rejected.

    Raises:
        DownloadError: on an unknown parameter or a repeated/empty value.
    """
    if not query:
        return {}, {}
    session_kwargs: dict[str, Any] = {}
    client_kwargs: dict[str, Any] = {}
    for key, values in parse_qs(query, keep_blank_values=True).items():
        if len(values) != 1 or not values[0]:
            raise DownloadError(f"s3 locator parameter {key!r} must have exactly one value", resumable=False)
        if key == "endpoint_url":
            client_kwargs["endpoint_url"] = values[0]
        elif key == "region":
            client_kwargs["region_name"] = values[0]
        elif key == "profile":
            session_kwargs["profile_name"] = values[0]
        else:
            raise DownloadError(f"unsupported s3 locator parameter {key!r}", resumable=False)
    return session_kwargs, client_kwargs


def _s3_open(url: str, have: int, state: dict[str, Any] | None, source_id: str) -> _Opened:
    # URL shape is validated BEFORE the lazy import: a malformed s3:// URL is
    # malformed even where boto3 exists, and must not be masked by its absence
    parts = _split_locator(url)
    bucket, key = parts.netloc, parts.path.lstrip("/")
    if not bucket or not key:
        raise DownloadError("malformed s3 chunk URL (missing bucket or key)", resumable=False)
    session_kwargs, client_kwargs = s3_locator_params(parts.query)
    try:
        import boto3  # noqa: PLC0415 — optional dependency; the base runtime never installs or imports it
    except ModuleNotFoundError as e:
        raise DownloadError(
            "s3 chunk transfers need boto3, which pithos never installs: "
            "pip install boto3 (or pluralis-pithos[s3]) in your own environment",
            resumable=False,
        ) from e
    label = _safe_label(url, source_id)
    kwargs: dict[str, Any] = {"Bucket": bucket, "Key": key}
    if have > 0:
        kwargs["Range"] = f"bytes={have}-"
        if state and state.get("etag"):
            kwargs["IfMatch"] = state["etag"]
    try:
        # A fresh Session per open, never the process-wide default session:
        # that one caches the first credentials it resolves (e.g. an instance
        # role primed by an unrelated library) and would silently override
        # env-provided keys for S3-compatible endpoints.
        client = boto3.session.Session(**session_kwargs).client("s3", **client_kwargs)
    except Exception as e:
        # SDK construction errors embed the offending values (a missing
        # profile's name, a malformed endpoint) — suppress the cause so the
        # cause chain cannot resurface them in tracebacks.
        raise DownloadError(
            f"cannot construct s3 client for {label} "
            f"({type(e).__name__}: check the locator's endpoint_url/region/profile parameters)",
            resumable=False,
        ) from None
    try:
        resp = client.get_object(**kwargs)
    except Exception as e:
        client.close()
        code = getattr(e, "response", {}).get("Error", {}).get("Code", "") if hasattr(e, "response") else ""
        if code in {"412", "416", "PreconditionFailed", "InvalidRange"}:
            raise _RangeStaleError() from e
        # the SDK error text can embed bucket/key — forward only its code/type
        raise DownloadError(f"GET {label} failed: {code or type(e).__name__}", resumable=have > 0) from e
    status = resp.get("ResponseMetadata", {}).get("HTTPStatusCode")
    last_modified = resp.get("LastModified")

    def close(body: Any = resp["Body"], client: Any = client) -> None:
        # body first, then the client's connection pool: a fresh client per
        # transfer must not leave its sockets to nondeterministic GC
        try:
            body.close()
        finally:
            close_client = getattr(client, "close", None)
            if close_client is not None:
                close_client()

    return _Opened(
        stream=resp["Body"],
        status=int(status) if status is not None else None,
        etag=resp.get("ETag"),
        last_modified=last_modified.isoformat() if last_modified is not None else None,
        range_start=_content_range_start(resp.get("ContentRange")),
        close=close,
    )


# ---------------------------------------------------------------- transfer
def _rebuild_hasher(fd: int, have: int, prefix_sha256: str | None, label: str) -> Any:
    """Re-derive the running sha256 over the retained prefix from the temp
    file itself and prove it still matches the sidecar — the partial is
    never trusted on the journal's word alone."""
    if prefix_sha256 is None:
        raise DownloadError(f"resume of {label} has no verified prefix hash", resumable=False)
    h = hashlib.sha256()
    off = 0
    while off < have:
        block = os.pread(fd, min(_BLOCK, have - off), off)
        if not block:
            break
        h.update(block)
        off += len(block)
    if off != have or h.hexdigest() != prefix_sha256:
        raise DownloadError(f"resume prefix of {label} changed under the lock — refusing to continue", resumable=False)
    return h


def _pump(
    opened: _Opened,
    label: str,
    fd: int,
    offset: int,
    expected: int,
    state_path: str,
    source_id: str,
    hasher: Any,
    etag: str | None,
    last_modified: str | None,
) -> None:
    """Stream the body into fd at `offset`, journaling every block; a cut
    stream keeps its vouched prefix (resumable), a short object raises with
    'failed integrity' (the pinned truncation contract). Never persists
    more than `expected` bytes: each read is capped one byte past the
    remaining length so an oversized body is rejected BEFORE the crossing
    block is written, and a short pwrite is looped to completion rather
    than assumed away."""
    while True:
        remaining = expected - offset
        try:
            block = opened.stream.read(min(_BLOCK, remaining + 1))
        except Exception as e:
            raise DownloadError(
                f"download of {label} cut at {offset} bytes: {type(e).__name__}", resumable=offset > 0
            ) from e
        if not block:
            break
        if len(block) > remaining:
            raise DownloadError(
                f"downloaded object exceeds {expected} bytes — failed integrity "
                "verification; resumable partial retained",
                resumable=offset > 0,
            )
        view = memoryview(block)
        while view:
            try:
                written = os.pwrite(fd, view, offset)
            except OSError as e:
                reason = e.strerror or type(e).__name__
                raise DownloadError(
                    f"cannot write download of {label} at byte {offset}: {reason}",
                    resumable=offset > 0,
                ) from e
            if written <= 0:
                raise DownloadError(
                    f"download of {label} cut at {offset} bytes: pwrite returned {written}", resumable=offset > 0
                )
            offset += written
            view = view[written:]
        hasher.update(block)
        save_state(
            state_path,
            {
                "bytes": offset,
                "sha256": hasher.hexdigest(),
                "source_id": source_id,
                "etag": etag,
                "last_modified": last_modified,
            },
        )
    if offset != expected:
        raise DownloadError(
            f"downloaded object truncated at {offset} of {expected} bytes — failed integrity "
            "verification; resumable partial retained",
            resumable=offset > 0,
        )


def _fetch_ranged(
    opener: Callable[[str, int, dict[str, Any] | None, str], _Opened],
    url: str,
    fd: int,
    have: int,
    expected: int,
    state_path: str,
    source_id: str,
    prefix_sha256: str | None,
) -> None:
    """Shared Range/If-Range resume state machine for urllib and S3."""
    state = load_state(state_path) if have > 0 else None
    label = _safe_label(url, source_id)
    attempts_left = 2  # at most one plain re-GET after a stale/misaligned range
    while True:
        try:
            opened = opener(url, have, state, source_id)
        except _RangeStaleError:
            have, state = 0, None
            attempts_left -= 1
            if attempts_left == 0:
                raise DownloadError(f"GET {label}: server repeatedly invalidates the range", resumable=False) from None
            continue
        if have == 0:
            break  # plain full-body GET
        if opened.status == 206 and opened.range_start == have:
            break  # aligned partial content: append
        if opened.status == 206:  # misaligned partial: cannot splice, re-GET
            opened.close()
            have, state = 0, None
            attempts_left -= 1
            if attempts_left == 0:
                raise DownloadError(
                    f"GET {label}: server keeps answering misaligned ranges", resumable=False
                ) from None
            continue
        # range ignored: this body already IS the full object from byte zero
        have = 0
        break
    try:
        if have > 0:
            hasher = _rebuild_hasher(fd, have, prefix_sha256, label)
            etag = opened.etag or (state or {}).get("etag")
            last_modified = opened.last_modified or (state or {}).get("last_modified")
        else:
            os.ftruncate(fd, 0)
            hasher = hashlib.sha256()
            etag, last_modified = opened.etag, opened.last_modified
            save_state(
                state_path,
                {
                    "bytes": 0,
                    "sha256": hasher.hexdigest(),
                    "source_id": source_id,
                    "etag": etag,
                    "last_modified": last_modified,
                },
            )
        _pump(opened, label, fd, have, expected, state_path, source_id, hasher, etag, last_modified)
    finally:
        opened.close()


def _fetch_local(
    path: str, fd: int, have: int, expected: int, state_path: str, source_id: str, prefix_sha256: str | None
) -> None:
    """Plain local source: resume by seek; (ino, size, mtime_ns) is the
    change stamp, so a source replaced mid-resume restarts from byte zero."""
    label = _safe_label(path, source_id)
    try:
        st = os.stat(path)
    except OSError as e:
        # OSError text embeds the filename — forward only the errno string
        raise DownloadError(f"cannot stat {label}: {e.strerror or type(e).__name__}", resumable=have > 0) from e
    stamp = f"{st.st_ino}:{st.st_size}:{st.st_mtime_ns}"
    state = load_state(state_path) if have > 0 else None
    if have > 0 and (state is None or state.get("etag") != stamp):
        have = 0
    try:
        src = open(path, "rb")
    except OSError as e:
        raise DownloadError(f"cannot open {label}: {e.strerror or type(e).__name__}", resumable=have > 0) from e
    with src:
        if have > 0:
            hasher = _rebuild_hasher(fd, have, prefix_sha256, label)
        else:
            os.ftruncate(fd, 0)
            hasher = hashlib.sha256()
        src.seek(have)
        _pump(
            _Opened(src, None, stamp, None, None, src.close),
            label,
            fd,
            have,
            expected,
            state_path,
            source_id,
            hasher,
            stamp,
            None,
        )


def fetch(
    url: str,
    fd: int,
    have: int,
    expected: int,
    state_path: str,
    source_id: str,
    prefix_sha256: str | None = None,
) -> None:
    """Complete the `expected`-byte object `url` in the temp descriptor `fd`,
    which already holds `have` verified prefix bytes (sha256 `prefix_sha256`).

    Returns only when fd holds exactly `expected` bytes (integrity vs the
    manifest is the caller's separate check). Journals resume state to
    `state_path`, keyed by `source_id` (the manifest-object identity from
    `object_identity`), never by locator; on a resumable failure the partial
    + journal stay for the next attempt, on any other failure the caller
    wipes both.

    Raises:
        DownloadError: on an unsupported/malformed URL, a transport error,
            a stale or repeatedly misaligned range, or a truncated object.
    """
    if have == expected:
        return
    if not 0 <= have < expected:
        raise DownloadError(f"resume offset {have} outside [0, {expected})", resumable=False)
    scheme = _split_locator(url).scheme.lower()
    if scheme == "":
        _fetch_local(url, fd, have, expected, state_path, source_id, prefix_sha256)
    elif scheme in ("http", "https", "file"):
        _fetch_ranged(_urllib_open, url, fd, have, expected, state_path, source_id, prefix_sha256)
    elif scheme == "s3":
        _fetch_ranged(_s3_open, url, fd, have, expected, state_path, source_id, prefix_sha256)
    else:
        raise DownloadError(f"unsupported chunk URL scheme {scheme!r}", resumable=False)


def read_small(url: str, max_bytes: int = 1 << 26) -> bytes:
    """A small whole object (e.g. a manifest.json) from any supported
    scheme, size-capped. S3 keeps the lazy boto3 import.

    Raises:
        DownloadError: on any transport failure or an oversized object.
    """
    scheme = _split_locator(url).scheme.lower()
    label = _safe_label(url, None)
    if scheme == "":
        try:
            with open(url, "rb") as f:
                data = f.read(max_bytes + 1)
        except OSError as e:
            raise DownloadError(f"cannot read {label}: {e.strerror or type(e).__name__}", resumable=False) from e
    elif scheme in ("http", "https", "file"):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(request, timeout=_URL_TIMEOUT) as resp:
                data = resp.read(max_bytes + 1)
        except Exception as e:
            raise DownloadError(f"GET {label} failed: {type(e).__name__}", resumable=False) from e
    elif scheme == "s3":
        opened = _s3_open(url, 0, None, "")
        try:
            data = opened.stream.read(max_bytes + 1)
        finally:
            opened.close()
    else:
        raise DownloadError(f"unsupported URL scheme {scheme!r}", resumable=False)
    if len(data) > max_bytes:
        raise DownloadError(f"{label} exceeds the {max_bytes}-byte small-object cap", resumable=False)
    return data
