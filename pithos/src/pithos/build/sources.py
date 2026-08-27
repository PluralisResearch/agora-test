"""Source adapters: pinned inventory + bounded-memory document iteration.

Supported sources: pinned Hugging Face Parquet, local Parquet/JSONL,
S3/R2-compatible object stores, immutable HTTP, and locked legacy binary
streams (byte-preserving re-chunk input). The lock's per-object inventory
(name, size, sha256) is authoritative; every object is verified against its
locked sha256 when read.

Memory and bandwidth are bounded: plan-time row evidence is metadata-only
(Parquet footer via ranged reads — never a full object download), workers
spool ONE source object at a time to a collision-free local file (the HF
hub cache is never used for source data), JSONL is parsed line by line,
and Parquet is iterated in row batches. Local files are hashed from the
same descriptor that is then parsed — no hash-then-reopen window.

Parquet reading uses pyarrow, imported lazily: it is a build-extra
dependency, never required by the base runtime.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import struct
import tempfile

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, BinaryIO, Protocol

from ..errors import BuildError
from . import policy as pol
from .recipe import Recipe
from .storage import HttpStorage, LocalStorage, S3Storage, Storage, storage_for


INVENTORY_SUFFIXES = (".jsonl", ".parquet")  # the one source-inventory rule for local/s3 (lock AND plan)


@dataclass(frozen=True)
class SourceItem:
    """One locked source object: storage-relative name, byte size, and the
    lock-pinned content sha256 (verified at read time)."""

    name: str
    size: int
    sha256: str

    @classmethod
    def from_lock(cls, raw: dict[str, Any]) -> SourceItem:
        return cls(name=str(raw["name"]), size=int(raw.get("size", 0)), sha256=str(raw["sha256"]))


def items_from_lock(lock_source: dict[str, Any]) -> list[SourceItem]:
    """The locked inventory as SourceItems (order preserved — it matters for
    legacy_binary; it is name-sorted for every other kind)."""
    return [SourceItem.from_lock(i) for i in lock_source["items"]]


def _verify_or_raise(item: SourceItem, digest: str) -> None:
    if digest != item.sha256:
        raise BuildError(
            f"source item {item.name!r} content mismatch: sha256 {digest} != locked {item.sha256} — refusing to build"
        )


def hash_file(path: str) -> str:
    """Streaming sha256 of a local file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def spool_stream(src: BinaryIO, item: SourceItem, spool_dir: str) -> str:
    """Stream one source object to the local spool, hashing and counting
    incrementally. Publication is atomic create-once under a CONTENT-derived
    name (collision-free by construction — 'a/b' and 'a__b' never alias):
    a private temp file is linked into place; a loser of that race (or a
    retry finding a survivor) hash-verifies the existing file and reuses it.
    A size/sha256 mismatch against the lock fails loudly; the temp file is
    always reclaimed."""
    os.makedirs(spool_dir, exist_ok=True)
    final = os.path.join(spool_dir, f"spool-{item.sha256}.bin")
    if os.path.exists(final):
        if hash_file(final) == item.sha256:
            return final  # verified survivor of an earlier attempt — skip the re-stream
        raise BuildError(f"spool file for {item.name!r} fails the locked sha256 — remove {final} and retry")
    fd, tmp = tempfile.mkstemp(dir=spool_dir, prefix=".spooling-")
    try:
        h = hashlib.sha256()
        n = 0
        with os.fdopen(fd, "wb") as dst:
            while chunk := src.read(1 << 20):
                h.update(chunk)
                n += len(chunk)
                dst.write(chunk)
        _verify_or_raise(item, h.hexdigest())
        if n != item.size:
            raise BuildError(f"source item {item.name!r} streamed {n} bytes, lock pinned {item.size}")
        try:
            os.link(tmp, final)  # atomic create-once: a concurrent spooler may have won
        except FileExistsError:
            if hash_file(final) != item.sha256:
                raise BuildError(
                    f"spool file for {item.name!r} exists but fails the locked sha256 — remove {final} and retry"
                ) from None
        return final
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@dataclass
class ItemHandle:
    """A locked source object verified and ready to parse: an open file
    positioned at 0 whose bytes were just verified against the lock. Local
    items are hashed from THIS descriptor (no hash-then-reopen window);
    remote items are parsed from the engine-owned spool file, which the
    caller must reclaim (delete `spooled`) after closing."""

    path: str  # local path, for diagnostics
    file: BinaryIO
    spooled: str | None  # engine-owned spool file to reclaim, else None

    def close(self) -> None:
        self.file.close()


def open_verified_local(path: str, item: SourceItem) -> ItemHandle:
    """Open one local source object and verify it against the lock through
    the SAME descriptor that parsing will use — no-follow on the final
    component, regular-file check, size + hash pass, rewind. There is no
    hash-then-reopen window for a swapped file to hide in."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise BuildError(f"source item {item.name!r} is not a regular file")
        if st.st_size != item.size:
            raise BuildError(f"source item {item.name!r} has {st.st_size} bytes, lock pinned {item.size}")
        h = hashlib.sha256()
        while chunk := os.read(fd, 1 << 20):
            h.update(chunk)
        _verify_or_raise(item, h.hexdigest())
        os.lseek(fd, 0, os.SEEK_SET)
        return ItemHandle(path=path, file=os.fdopen(fd, "rb"), spooled=None)
    except BaseException:
        os.close(fd)
        raise


def _parse_jsonl(f: BinaryIO, item: SourceItem) -> Iterator[tuple[str, str, str]]:
    crawl = pol.crawl_of(item.name)
    text = io.TextIOWrapper(f, encoding="utf-8")
    for lineno, line in enumerate(text):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            doc_id = rec["id"]
            body = rec["text"]
        except (json.JSONDecodeError, KeyError) as e:
            raise BuildError(f"{item.name}:{lineno + 1}: malformed JSONL document: {e}") from e
        if not isinstance(doc_id, str) or not isinstance(body, str):
            raise BuildError(f"{item.name}:{lineno + 1}: 'id' and 'text' must be strings")
        doc_crawl = rec.get("crawl") or rec.get("dump") or crawl
        yield str(doc_crawl), doc_id, body


def _parse_parquet(f: BinaryIO, item: SourceItem, batch_size: int = 512) -> Iterator[tuple[str, str, str]]:
    """Row-batch iteration over an open parquet stream — bounded memory."""
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415 — optional build dependency
    except ImportError as e:
        raise BuildError("parquet sources need the 'build' extra (pyarrow)") from e
    try:
        pf = pq.ParquetFile(f)
    except Exception as e:
        raise BuildError(f"{item.name}: cannot open parquet source: {e}") from e
    missing = sorted({"id", "text"} - set(pf.schema_arrow.names))
    if missing:
        raise BuildError(f"{item.name}: parquet source missing required columns: {', '.join(missing)}")
    crawl = pol.crawl_of(item.name)
    has_dump = "dump" in pf.schema_arrow.names
    columns = ["id", "text", "dump"] if has_dump else ["id", "text"]
    try:
        for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
            ids = batch.column("id").to_pylist()
            texts = batch.column("text").to_pylist()
            dumps = batch.column("dump").to_pylist() if has_dump else [None] * len(ids)
            for doc_id, body, dump in zip(ids, texts, dumps, strict=True):
                if not isinstance(doc_id, str) or not isinstance(body, str):
                    raise BuildError(f"{item.name}: parquet 'id'/'text' columns must be strings")
                yield str(dump) if dump else crawl, doc_id, body
    except BuildError:
        raise
    except Exception as e:
        raise BuildError(f"{item.name}: cannot read parquet source: {e}") from e


def parse_item(item: SourceItem, f: BinaryIO) -> Iterator[tuple[str, str, str]]:
    """Stream (crawl, document_id, text) documents from a verified item."""
    if item.name.endswith(".jsonl"):
        yield from _parse_jsonl(f, item)
    elif item.name.endswith(".parquet"):
        yield from _parse_parquet(f, item)
    else:
        raise BuildError(f"unsupported source item {item.name!r} (want .jsonl or .parquet)")


def _footer_rows_ranged(size: int, read_at: Callable[[int, int], bytes], what: str) -> int:
    """Parquet footer row count via TWO small ranged reads (tail, then
    footer), METADATA-ONLY. The footer is reassembled into a minimal parquet
    container for pyarrow — pyarrow reads Python streams whole, so handing
    it a raw ranged view would pull the entire object."""
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415 — optional build dependency
    except ImportError as e:
        raise BuildError("parquet row evidence needs the 'build' extra (pyarrow)") from e
    if size < 12:
        raise BuildError(f"cannot read the parquet footer of {what} for row evidence: {size} bytes")
    tail = read_at(size - 8, 8)
    if len(tail) != 8 or tail[4:] != b"PAR1":
        raise BuildError(f"cannot read the parquet footer of {what} for row evidence: bad tail magic")
    (footer_len,) = struct.unpack("<i", tail[:4])
    if not 0 < footer_len <= size - 12:
        raise BuildError(f"cannot read the parquet footer of {what} for row evidence: bad footer length {footer_len}")
    footer = read_at(size - 8 - footer_len, footer_len)
    if len(footer) != footer_len:
        raise BuildError(f"cannot read the parquet footer of {what} for row evidence: short ranged read")
    try:
        return int(pq.ParquetFile(io.BytesIO(b"PAR1" + footer + tail)).metadata.num_rows)
    except Exception as e:
        raise BuildError(f"cannot read the parquet footer of {what} for row evidence: {e}") from e


def _stream_read_at(f: BinaryIO, start: int, n: int) -> bytes:
    f.seek(start)
    return f.read(n)


def declared_rows(storage: Storage, item: SourceItem) -> int | None:
    """Parquet footer row count for plan-time reconciliation, METADATA-ONLY:
    two small ranged reads (local files seek in place; S3/R2 via ranged
    GETs). Non-parquet items carry no row evidence (None)."""
    if not item.name.endswith(".parquet"):
        return None
    if isinstance(storage, LocalStorage):
        with open(storage.contained_path(item.name), "rb") as f:
            return _footer_rows_ranged(item.size, lambda s, n: _stream_read_at(f, s, n), item.name)
    if isinstance(storage, S3Storage):
        return _footer_rows_ranged(item.size, lambda s, n: storage.read_range(item.name, s, n), item.name)
    return None


class SourceAdapter(Protocol):
    def open_item(self, item: SourceItem, spool_dir: str) -> ItemHandle:
        """Make one locked item available as a verified local stream."""
        ...

    def iter_documents(self, item: SourceItem, handle: ItemHandle) -> Iterator[tuple[str, str, str]]: ...


class StorageSource:
    """Source over any listable Storage (local dirs, S3/R2). Covers the
    'local' and 's3' recipe source kinds."""

    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def open_item(self, item: SourceItem, spool_dir: str) -> ItemHandle:
        if isinstance(self.storage, LocalStorage):
            return open_verified_local(self.storage.contained_path(item.name), item)
        with self.storage.open_read(item.name) as src:
            path = spool_stream(src, item, spool_dir)
        return ItemHandle(path=path, file=open(path, "rb"), spooled=path)

    def iter_documents(self, item: SourceItem, handle: ItemHandle) -> Iterator[tuple[str, str, str]]:
        yield from parse_item(item, handle.file)


class HubClient(Protocol):
    """The slice of the Hugging Face hub API pithos uses, with the REAL
    huggingface_hub signatures — fakes must match these exactly."""

    def dataset_info(self, repo_id: str, *, revision: str | None = None) -> Any:
        """-> object with .sha (the resolved commit)."""
        ...

    def model_info(self, repo_id: str, *, revision: str | None = None) -> Any:
        """-> object with .sha (tokenizer repos are model repos)."""
        ...

    def list_repo_tree(
        self, repo_id: str, *, repo_type: str, revision: str, recursive: bool, expand: bool
    ) -> Iterable[Any]:
        """-> RepoFile-shaped objects: .path, .size, .lfs (None or a
        BlobLfsInfo-shaped object with .sha256 — the LFS content digest)."""
        ...

    def hf_hub_download(self, repo_id: str, filename: str, *, repo_type: str, revision: str) -> str:
        """-> local path of the pinned-revision file (small assets only —
        tokenizer.json; NEVER source data objects)."""
        ...

    def open_ranged(self, repo_id: str, filename: str, *, repo_type: str, revision: str) -> BinaryIO:
        """-> a seekable read stream of the pinned-revision object (the real
        client uses HfFileSystem: ranged HTTP, no hub-cache accumulation).
        Used for footer metadata reads and for streamed worker spooling."""
        ...


class HfHubClient:
    """Real hub client: delegates to huggingface_hub (lazy import — a
    build-extra dependency)."""

    def __init__(self) -> None:
        try:
            import huggingface_hub  # noqa: PLC0415 — optional build dependency
        except ImportError as e:
            raise BuildError("huggingface sources need the 'build' extra (huggingface_hub)") from e
        self._hub = huggingface_hub
        self._api = huggingface_hub.HfApi()
        self._fs: Any = None

    def dataset_info(self, repo_id: str, *, revision: str | None = None) -> Any:
        return self._api.dataset_info(repo_id, revision=revision)

    def model_info(self, repo_id: str, *, revision: str | None = None) -> Any:
        return self._api.model_info(repo_id, revision=revision)

    def list_repo_tree(
        self, repo_id: str, *, repo_type: str, revision: str, recursive: bool, expand: bool
    ) -> Iterable[Any]:
        return self._api.list_repo_tree(
            repo_id, repo_type=repo_type, revision=revision, recursive=recursive, expand=expand
        )

    def hf_hub_download(self, repo_id: str, filename: str, *, repo_type: str, revision: str) -> str:
        return self._hub.hf_hub_download(repo_id, filename, repo_type=repo_type, revision=revision)

    def open_ranged(self, repo_id: str, filename: str, *, repo_type: str, revision: str) -> BinaryIO:
        if self._fs is None:
            self._fs = self._hub.HfFileSystem()
        plural = "datasets" if repo_type == "dataset" else "models"
        return self._fs.open(f"{plural}/{repo_id}@{revision}/{filename}", "rb")


def hf_locked_inventory(
    client: HubClient, dataset: str, revision: str, include: tuple[str, ...], exclude: tuple[str, ...]
) -> tuple[str, list[dict[str, Any]]]:
    """Lock-time HF resolution: pin the exact commit and the SELECTED file
    inventory with LFS content digests (BlobLfsInfo.sha256). Fails clearly
    when a selected object has no content digest — a path is never accepted
    as a digest."""
    try:
        commit = str(client.dataset_info(dataset, revision=revision).sha)
    except Exception as e:
        raise BuildError(f"cannot resolve {dataset!r} revision {revision!r}: {e}") from e
    try:
        tree = client.list_repo_tree(dataset, repo_type="dataset", revision=commit, recursive=True, expand=True)
    except Exception as e:
        raise BuildError(f"cannot list {dataset!r} at {commit}: {e}") from e
    items: list[dict[str, Any]] = []
    for entry in tree:
        path = str(getattr(entry, "path", ""))
        if not path.endswith(".parquet") or not pol.selected(path, include, exclude):
            continue
        lfs = getattr(entry, "lfs", None)
        digest = getattr(lfs, "sha256", None) if lfs is not None else None
        if not isinstance(digest, str) or len(digest) != 64:
            raise BuildError(f"hf object {path!r} has no LFS sha256 — cannot pin content evidence")
        items.append({"name": path, "size": int(getattr(entry, "size", 0)), "sha256": digest})
    if not items:
        raise BuildError(f"hf source {dataset!r}: selection is empty at {commit}")
    return commit, sorted(items, key=lambda i: i["name"])


def hf_declared_rows(client: HubClient, dataset: str, revision: str, item: SourceItem) -> int | None:
    """Parquet footer row count over pinned-revision RANGED reads — the plan
    scan never downloads source objects. Fails loudly when the footer of a
    parquet item cannot be read (reconciliation evidence must be real)."""
    if not item.name.endswith(".parquet"):
        return None
    try:
        f = client.open_ranged(dataset, item.name, repo_type="dataset", revision=revision)
    except Exception as e:
        raise BuildError(f"cannot open {dataset!r}:{item.name} at {revision} for row evidence: {e}") from e
    with f:
        return _footer_rows_ranged(item.size, lambda s, n: _stream_read_at(f, s, n), item.name)


class HuggingFaceParquetSource:
    """Pinned Hugging Face Parquet source. Builds ONLY from the locked
    inventory; workers stream ONE object at a time over pinned-revision
    ranged reads into a collision-free local spool (the hub cache is never
    used for source data) and verify the locked LFS sha256 while spooling."""

    def __init__(self, dataset: str, revision: str, client: HubClient) -> None:
        self.dataset = dataset
        self.revision = revision
        self.client = client

    def open_item(self, item: SourceItem, spool_dir: str) -> ItemHandle:
        try:
            src = self.client.open_ranged(self.dataset, item.name, repo_type="dataset", revision=self.revision)
        except Exception as e:
            raise BuildError(f"cannot open {self.dataset!r}:{item.name} at {self.revision}: {e}") from e
        with src:
            path = spool_stream(src, item, spool_dir)
        return ItemHandle(path=path, file=open(path, "rb"), spooled=path)

    def iter_documents(self, item: SourceItem, handle: ItemHandle) -> Iterator[tuple[str, str, str]]:
        yield from parse_item(item, handle.file)


class HttpSource:
    """Immutable HTTP source over recipe-declared items with lock-pinned
    sha256 digests (verified on every read)."""

    def __init__(self, storage: HttpStorage) -> None:
        self.storage = storage

    def open_item(self, item: SourceItem, spool_dir: str) -> ItemHandle:
        with self.storage.open_read(item.name) as src:
            path = spool_stream(src, item, spool_dir)
        return ItemHandle(path=path, file=open(path, "rb"), spooled=path)

    def iter_documents(self, item: SourceItem, handle: ItemHandle) -> Iterator[tuple[str, str, str]]:
        yield from parse_item(item, handle.file)


def adapter_for(
    recipe: Recipe, lock_source: dict[str, Any], *, client: Any = None, fetcher: Any = None
) -> SourceAdapter:
    """Construct the source adapter for a locked recipe. The lock's source
    evidence is authoritative; mismatched kinds fail loudly."""
    kind = recipe.source["kind"]
    if lock_source.get("kind") != kind:
        raise BuildError(f"lock source kind {lock_source.get('kind')!r} != recipe source kind {kind!r}")
    if kind == "huggingface":
        return HuggingFaceParquetSource(
            recipe.source["dataset"], lock_source["revision"], client if client is not None else HfHubClient()
        )
    if kind == "http":
        storage = storage_for(recipe.source["base_url"], fetcher=fetcher)
        if not isinstance(storage, HttpStorage):
            raise BuildError(f"http base_url {recipe.source['base_url']!r} did not resolve to HTTP storage")
        return HttpSource(storage)
    storage = storage_for(recipe.source["uri"], client=client)
    if kind == "s3" and not isinstance(storage, S3Storage):
        raise BuildError(f"s3 source uri {recipe.source['uri']!r} did not resolve to S3 storage")
    return StorageSource(storage)
