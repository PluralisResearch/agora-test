"""Storage abstraction for the build plane.

Every build artifact and every source object is addressed through a
`Storage`, so the engine is identical against a local directory, an S3/R2
bucket, or an immutable HTTP origin. Large objects (source items, parts,
chunks) are ALWAYS streamed via open_read/open_write — read_bytes is for
small control objects (plans, markers, manifests) only.

Create-once publication is atomic, not check-then-act: local writes publish
via hard-link (fails if the name exists), S3/R2 via conditional put
(If-None-Match: *). A conflict raises ImmutableConflict; callers then verify
the existing object's hash instead of overwriting.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.parse

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Any, BinaryIO, Protocol, cast

from ..errors import BuildError


class ImmutableConflict(BuildError):  # noqa: N818 — raised for create-once races and always caught/verified by the caller; it is a control-flow signal, not a failure
    """A create-once write found an existing object under the same name."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"object {name!r} already exists — verify it, never blindly overwrite")


@dataclass(frozen=True)
class ObjectInfo:
    """One stored object: name (storage-relative), size in bytes, and the
    store's version evidence (etag / version id / content hash)."""

    name: str
    size: int
    etag: str


def _hash_stream(f: BinaryIO) -> str:
    h = hashlib.sha256()
    while chunk := f.read(1 << 20):
        h.update(chunk)
    return h.hexdigest()


def check_object_name(name: Any, what: str) -> str:
    """The ONE canonical safe storage-relative POSIX object-name rule,
    enforced at every recipe/lock/storage boundary: a non-empty relative
    path with no empty/dot/dot-dot segments, no backslashes, and no NUL or
    control characters — so S3 and HTTP stores get the same traversal
    protection LocalStorage's containment gives local names, and no name
    can change meaning when encoded into a URL. Nested paths such as
    'data/CC-MAIN-2024-10/000_00000.parquet' remain valid."""
    if not isinstance(name, str) or not name:
        raise BuildError(f"{what}: object name must be a non-empty string")
    if name.startswith("/"):
        raise BuildError(f"{what}: absolute object name {name!r}")
    if "\\" in name:
        raise BuildError(f"{what}: backslash in object name {name!r}")
    if any(ord(c) < 0x20 or ord(c) == 0x7F for c in name):
        raise BuildError(f"{what}: control character in object name {name!r}")
    if any(seg in {"", ".", ".."} for seg in name.split("/")):
        raise BuildError(f"{what}: empty/dot/dot-dot segment in object name {name!r}")
    return name


def check_inventory_names(names: Any, what: str) -> None:
    """check_object_name over a whole inventory plus duplicate rejection."""
    if not isinstance(names, (list, tuple)):
        raise BuildError(f"{what}: inventory names must be a list")
    seen: set[str] = set()
    for n in names:
        check_object_name(n, what)
        if n in seen:
            raise BuildError(f"{what}: duplicate object name {n!r}")
        seen.add(n)


class Storage(Protocol):
    """Object-store contract used by the build engine."""

    def list(self, prefix: str) -> list[ObjectInfo]:
        """All objects under `prefix`, sorted by name (deterministic)."""
        ...

    def read_bytes(self, name: str) -> bytes:
        """Small control objects only — never source items, parts, or chunks."""
        ...

    def write_bytes(self, name: str, data: bytes, *, create_once: bool = False) -> None: ...

    def open_read(self, name: str) -> BinaryIO: ...

    def open_write(self, name: str, *, create_once: bool = False) -> AbstractContextManager[BinaryIO]:
        """Streamed write, published atomically on close. With create_once,
        publication raises ImmutableConflict if the name exists."""
        ...

    def exists(self, name: str) -> bool: ...

    def sha256(self, name: str) -> str:
        """Streaming content hash of a stored object."""
        ...


class LocalStorage:
    """Storage rooted at a local directory. Names are validated as relative
    and contained; the realpath of the existing ancestor chain must stay
    inside the root, so symlinked parents cannot escape. Final-component
    symlinks are rejected for reads and never followed for writes."""

    def __init__(self, root: str) -> None:
        self.root = os.path.realpath(root)
        os.makedirs(self.root, exist_ok=True)

    def _path(self, name: str) -> str:
        if os.path.isabs(name) or ".." in name.split("/"):
            raise BuildError(f"unsafe storage name {name!r}")
        path = os.path.join(self.root, name)
        real = os.path.realpath(path)  # resolves the existing ancestor chain
        if real != self.root and not real.startswith(self.root + os.sep):
            raise BuildError(f"storage name {name!r} escapes the root (symlinked parent?)")
        return path

    def list(self, prefix: str) -> list[ObjectInfo]:
        base = self._path(prefix) if prefix else self.root
        out: list[ObjectInfo] = []
        if not os.path.isdir(base):
            return out
        for dirpath, dirnames, filenames in os.walk(base):  # followlinks=False by default
            for d in list(dirnames):
                full = os.path.join(dirpath, d)
                if os.path.islink(full):
                    raise BuildError(f"unsafe symlinked directory in storage: {full!r}")
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                if os.path.islink(full):
                    raise BuildError(f"unsafe symlink in storage: {full!r}")
                rel = os.path.relpath(full, self.root)
                with open(full, "rb") as f:
                    etag = _hash_stream(f)
                out.append(ObjectInfo(rel, os.path.getsize(full), etag))
        out.sort(key=lambda o: o.name)
        return out

    def read_bytes(self, name: str) -> bytes:
        with self.open_read(name) as f:
            return f.read()

    def open_read(self, name: str) -> BinaryIO:
        path = self._path(name)
        if os.path.islink(path):
            raise BuildError(f"refusing to read through symlink {name!r}")
        return open(path, "rb")

    def write_bytes(self, name: str, data: bytes, *, create_once: bool = False) -> None:
        with self.open_write(name, create_once=create_once) as f:
            f.write(data)

    @contextmanager
    def open_write(self, name: str, *, create_once: bool = False) -> Iterator[BinaryIO]:
        path = self._path(name)
        parent = os.path.dirname(path)
        os.makedirs(parent, exist_ok=True)
        if os.path.islink(path):
            raise BuildError(f"refusing to write through symlink {name!r}")
        fd, tmp = tempfile.mkstemp(dir=parent, prefix=".spool-")
        try:
            with os.fdopen(fd, "wb") as f:
                yield f
                f.flush()
                os.fsync(f.fileno())
            if create_once:
                try:
                    os.link(tmp, path)  # atomic create-once: fails if the name exists
                except FileExistsError:
                    raise ImmutableConflict(name) from None
                os.unlink(tmp)
            else:
                os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def exists(self, name: str) -> bool:
        return os.path.isfile(self._path(name))

    def sha256(self, name: str) -> str:
        with self.open_read(name) as f:
            return _hash_stream(f)

    def contained_path(self, name: str) -> str:
        """The real local path of an object, for in-place footer reads —
        same containment and symlink checks as open_read."""
        path = self._path(name)
        if os.path.islink(path):
            raise BuildError(f"refusing to use symlink {name!r}")
        if not os.path.isfile(path):
            raise BuildError(f"no such object {name!r}")
        return path


class S3Storage:
    """S3/R2-compatible storage over a duck-typed client (boto3-compatible:
    list_objects_v2/get_object/put_object/head_object). The client is
    injected so tests use an in-memory fake — pithos never touches
    credentials itself. Streamed writes spool to a local temp file and are
    put on close; create-once uses the If-None-Match precondition."""

    def __init__(self, bucket: str, prefix: str, client: Any) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = client

    def _key(self, name: str) -> str:
        return f"{self.prefix}/{name}" if self.prefix else name

    def _safe(self, name: str) -> str:
        return check_object_name(name, f"s3://{self.bucket} object")

    @staticmethod
    def _is_not_found(e: Exception) -> bool:
        """True only for a clean NotFound (botocore ClientError code or the
        typed service exceptions); auth/service failures are NOT 'absent'."""
        code = getattr(e, "response", {}).get("Error", {}).get("Code", "") if hasattr(e, "response") else ""
        return code in {"404", "NoSuchKey", "NotFound", "NoSuchBucket"} or type(e).__name__ in {
            "NoSuchKey",
            "NotFound",
        }

    def list(self, prefix: str) -> list[ObjectInfo]:
        out: list[ObjectInfo] = []
        token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": self._key(prefix)}
            if token:
                kwargs["ContinuationToken"] = token
            try:
                resp = self.client.list_objects_v2(**kwargs)
            except Exception as e:
                raise BuildError(f"list s3://{self.bucket}/{self._key(prefix)} failed: {e}") from e
            for obj in resp.get("Contents", []):
                name = obj["Key"][len(self._key("")) :]
                out.append(ObjectInfo(name, obj["Size"], str(obj.get("ETag", "")).strip('"')))
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        out.sort(key=lambda o: o.name)
        return out

    def read_bytes(self, name: str) -> bytes:
        with self.open_read(name) as f:
            return f.read()

    def open_read(self, name: str) -> BinaryIO:
        self._safe(name)
        try:
            return self.client.get_object(Bucket=self.bucket, Key=self._key(name))["Body"]
        except Exception as e:
            if self._is_not_found(e):
                raise BuildError(f"s3://{self.bucket}/{self._key(name)} not found") from e
            raise BuildError(f"get s3://{self.bucket}/{self._key(name)} failed: {e}") from e

    def read_range(self, name: str, start: int, length: int) -> bytes:
        """One ranged GET (bytes=start..start+length-1) — footer metadata
        reads never pull the whole object."""
        self._safe(name)
        if start < 0 or length < 0:
            raise BuildError(f"invalid range {start}+{length} for {name!r}")
        try:
            body = self.client.get_object(
                Bucket=self.bucket, Key=self._key(name), Range=f"bytes={start}-{start + length - 1}"
            )["Body"]
            return body.read()
        except Exception as e:
            if self._is_not_found(e):
                raise BuildError(f"s3://{self.bucket}/{self._key(name)} not found") from e
            raise BuildError(f"range get s3://{self.bucket}/{self._key(name)} failed: {e}") from e

    def write_bytes(self, name: str, data: bytes, *, create_once: bool = False) -> None:
        with self.open_write(name, create_once=create_once) as f:
            f.write(data)

    @contextmanager
    def open_write(self, name: str, *, create_once: bool = False) -> Iterator[BinaryIO]:
        self._safe(name)
        spool = cast(BinaryIO, tempfile.SpooledTemporaryFile(max_size=64 << 20))  # RAM to 64 MiB, then disk
        try:
            yield spool
            spool.seek(0)
            kwargs: dict[str, Any] = {"Bucket": self.bucket, "Key": self._key(name), "Body": spool}
            if create_once:
                kwargs["IfNoneMatch"] = "*"
            try:
                self.client.put_object(**kwargs)
            except Exception as e:
                if create_once and self._is_precondition_failed(e):
                    raise ImmutableConflict(name) from e
                raise BuildError(f"put s3://{self.bucket}/{self._key(name)} failed: {e}") from e
        finally:
            spool.close()

    @staticmethod
    def _is_precondition_failed(e: Exception) -> bool:
        code = getattr(e, "response", {}).get("Error", {}).get("Code", "") if hasattr(e, "response") else ""
        return (
            code in {"PreconditionFailed", "412"}
            or "PreconditionFailed" in type(e).__name__
            or "PreconditionFailed" in str(e)
        )

    def exists(self, name: str) -> bool:
        self._safe(name)
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(name))
            return True
        except Exception as e:
            if self._is_not_found(e):
                return False
            raise BuildError(f"head s3://{self.bucket}/{self._key(name)} failed: {e}") from e

    def sha256(self, name: str) -> str:
        with self.open_read(name) as f:
            return _hash_stream(f)


def check_http_base_url(base_url: Any) -> str:
    """Coherent immutable-HTTP base URL: absolute http(s) with a host, no
    query/fragment, and no whitespace or control characters — anything else
    would make the joined object URLs ambiguous or unparseable."""
    if not isinstance(base_url, str) or not base_url:
        raise BuildError("http base URL must be a non-empty string")
    if any(ord(c) <= 0x20 or ord(c) == 0x7F for c in base_url):
        raise BuildError(f"http base URL {base_url!r} carries whitespace or control characters")
    parts = urllib.parse.urlsplit(base_url)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise BuildError(f"http base URL {base_url!r} must be an absolute http(s) URL with a host")
    if parts.query or parts.fragment:
        raise BuildError(f"http base URL {base_url!r} must not carry a query or fragment")
    return base_url


class HttpStorage:
    """Read-only storage over an immutable HTTP origin. The fetcher is
    injected (tests fake it, the CLI uses UrllibFetcher); writes fail.
    Object names are percent-encoded per path segment, so reserved
    characters (?, #, %) and non-ASCII bytes can never change the addressed
    resource or alias two distinct names onto one URL."""

    def __init__(self, base_url: str, fetcher: Any) -> None:
        self.base_url = check_http_base_url(base_url).rstrip("/")
        self.fetcher = fetcher  # .open(url) -> BinaryIO; .head(url) -> dict[str, str]

    def _url(self, name: str) -> str:
        check_object_name(name, f"http source {self.base_url} object")
        encoded = "/".join(urllib.parse.quote(seg, safe="") for seg in name.split("/"))
        return f"{self.base_url}/{encoded}"

    def list(self, prefix: str) -> list[ObjectInfo]:
        raise BuildError("HTTP origins are not listable — declare source items explicitly in the recipe")

    def read_bytes(self, name: str) -> bytes:
        with self.open_read(name) as f:
            return f.read()

    def open_read(self, name: str) -> BinaryIO:
        return self.fetcher.open(self._url(name))

    def write_bytes(self, name: str, data: bytes, *, create_once: bool = False) -> None:
        raise BuildError("HTTP origins are read-only sources, never build targets")

    def open_write(self, name: str, *, create_once: bool = False) -> BinaryIO:
        raise BuildError("HTTP origins are read-only sources, never build targets")

    def exists(self, name: str) -> bool:
        url = self._url(name)  # name safety is validated before any exception swallowing
        try:
            self.fetcher.head(url)
            return True
        except Exception:
            return False

    def sha256(self, name: str) -> str:
        with self.open_read(name) as f:
            return _hash_stream(f)


class UrllibFetcher:
    """Real HTTP fetcher for immutable sources (stdlib urllib; no secrets)."""

    def open(self, url: str) -> BinaryIO:
        import urllib.request  # noqa: PLC0415 — stdlib, lazy so import time stays clean

        try:
            return urllib.request.urlopen(url)  # noqa: S310 — recipe-declared immutable origins
        except Exception as e:
            raise BuildError(f"GET {url} failed: {e}") from e

    def head(self, url: str) -> dict[str, str]:
        import urllib.request  # noqa: PLC0415

        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as resp:  # noqa: S310
                return dict(resp.headers.items())
        except Exception as e:
            raise BuildError(f"HEAD {url} failed: {e}") from e


def storage_for(uri: str, *, client: Any = None, fetcher: Any = None, session_factory: Any = None) -> Storage:
    """Resolve a build/source URI to a Storage. `file://` or a plain path is
    local; `s3://bucket/prefix[?endpoint_url=...&profile=...]` needs an
    injected or boto3 client (endpoint/profile configure R2-style services
    via the environment's own config — never credentials in the URI);
    `http(s)://` needs a fetcher (UrllibFetcher by default)."""
    if uri.startswith("s3://"):
        rest = uri[len("s3://") :]
        authority, _, query = rest.partition("?")
        bucket, _, prefix = authority.partition("/")
        params = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
        if client is None:
            try:
                import boto3  # noqa: PLC0415 — optional build dependency
            except ImportError as e:
                raise BuildError("s3:// URIs need the 'build' extra (boto3) or an injected client") from e
            factory = session_factory or boto3.Session
            session = factory(profile_name=params["profile"]) if "profile" in params else factory()
            client = session.client("s3", endpoint_url=params.get("endpoint_url"))
        return S3Storage(bucket, prefix, client)
    if uri.startswith("http://") or uri.startswith("https://"):
        return HttpStorage(uri, fetcher if fetcher is not None else UrllibFetcher())
    if uri.startswith("file://"):
        uri = uri[len("file://") :]
    return LocalStorage(uri)
