# Provenance

Pithos's deterministic primitives and reference stream reader were imported
from **[pretrain-data](https://github.com/PluralisResearch/pretrain-data)**,
the reference data-prep repo extracted from the Colonnade training system.

| | |
|---|---|
| Source repo | `https://github.com/PluralisResearch/pretrain-data` |
| Original source tip | `d09a95529f919b879aae918878499084859e77bd` (2026-07-20, "Docs move to the repo wiki…") |
| Filtered tip | `f553c78d72024bfdba427efb3b37e15bf0a0ef5a` (same history after `git filter-repo --to-subdirectory-filter pithos`) |
| Import date | 2026-08-06 |
| Import method | The source tree at `d09a955` was imported as a **single squashed commit** (the upstream repo's own 10-commit history was deliberately not carried over; this table and the file-by-file mapping below are the provenance record). After that commit, `pithos/` is ordinary tracked content of this repository — **no** submodule, nested `.git` directory, or ongoing subtree dependency. |

## File-by-file mapping

| pretrain-data @ d09a955 | pithos | notes |
|---|---|---|
| `reader.py` → `StreamShard` | `src/pithos/torch.py` | Behavior-identical mapping (golden vectors unchanged); composed from `manifest.py` + `cache.py`. Returns `torch.LongTensor [rows, seq+1]`. Reference reader, not the final API. |
| `reader.py` (chunk fetch/mmap/eviction internals) | `src/pithos/cache.py` | `ChunkCache`, **hardened past the source** (see divergences). |
| `reader.py` (manifest validation + per-seq sample arithmetic) | `src/pithos/manifest.py` | `Manifest` + `StreamLayout`, **hardened past the source**; canonical `manifest_sha256` identity. |
| `prep_corpus.py` → `key_of`, `t1_from_fraction`, `write_record`, `read_records`, `tokenize_docs`, `ChunkWriter`, `merge_parts`, `numpy_sorted_records` | `src/pithos/corpus.py` | Byte-deterministic producer primitives, pure Python/NumPy, typed errors. Re-exported via `pithos.build`. |
| `prep_corpus.py` identity constants (dataset/revision/tokenizer/`HASH_SEED`/…) | `recipes/legacy_fineweb_edu_tranche1.snapshot.json` | Recorded as data, labeled unmistakably as an external build. |
| — (new in pithos) | `src/pithos/errors.py` | `PithosError` hierarchy; replaces the source's assert/SystemExit style. |
| `test_prep_corpus.py` | `tests/test_corpus_primitives.py` | Same assertions, minus the fleet user-data test (fleet not imported); SystemExit expectations updated to typed errors. |
| `test_reader_stream.py` | `tests/test_stream_reader.py` | Golden vectors unchanged — the legacy reader contract is pinned by the same digests. |

## Behavior contract

The golden vectors in `tests/test_stream_reader.py` pin
`(manifest, seq, seed, sample_id) → tokens` for the LEGACY
`colonnade_stream_v1` format. They are unchanged from the source repo, where
they were certified byte-identical to the Colonnade production reader
(`colonnade_worker/runtimes.py`) on 2026-07-20. A digest change is a NEW
reader contract and must ship as a new manifest format tag with its own
vectors alongside the old — never as a silent edit.

## Deliberate divergences from the source (milestone-1 reviews)

1. **Approved default contract**: new corpora use `pithos_stream_v1` — flat
   little-endian int32 (`<i4`) objects of EXACTLY 2²⁶ logical tokens
   (256 MiB) + one overlap token, enforced at manifest validation;
   `corpus.CHUNK_TOKENS` is `1 << 26` (source: `1 << 30`). The legacy 2³⁰
   format remains readable via explicit compatibility.
2. **Integrity is verified, not calculated**: manifests must carry a valid
   per-chunk sha256 and a self-consistent canonical identity hash; the cache
   verifies size + sha256 before publishing or mapping any object.
3. **Path safety**: chunk names are validated as plain relative basenames
   and every filesystem path is resolve-checked for containment in
   cache_dir; the `.locks` directory is validated as real and contained, and
   lock files are opened dir-relative with O_NOFOLLOW.
4. **Concurrency**: no startup sweep (valid cached objects survive
   restarts); per-chunk `fcntl.flock` interprocess locks; atomic temp-file
   publication; failed downloads are deleted and retried from byte zero
   (true HTTP range resume is a milestone-3 blocker, with prefetch).
5. **Cache semantics**: a byte-budgeted ON-DISK LRU (1 GiB default,
   configurable) over leased objects — recency persisted via mtimes and
   refreshed on hits, eviction under per-chunk locks, restart-aware
   accounting, explicit `<i4` mmaps, and a documented single-oversized
   -object exception (source: object-count `keep`, native-endian `np.int32`,
   no disk management).
6. **Typed errors**: `PithosError` hierarchy (a ValueError) replaces the
   source's `assert`/`SystemExit` in library code, with added argument
   validation (key length, record sizes, chunk_tokens, batch, negative token
   ids, part-buffer bounds).
7. **Supported seq set**: `pithos_stream_v1` allows exactly the powers of
   two 2048–131072; the legacy format keeps the imported any-divisor rule so
   the golden vectors stay valid.
8. **Reader argument naming**: `StreamShard.ids(batch_id, rows)` keeps the
   inherited legacy batch-index semantics (pinned by the golden vectors) and
   is documented as such; absolute sample-range starts matching lease
   allocation are the milestone-3 Corpus/PithosBatchSource API.

## Deliberately not imported (later milestones)

| pretrain-data file | why not in milestone 1 |
|---|---|
| `prep_corpus.py` S3/HF/fleet phases (`files`/`work`/`plan`/`merge`/`verify`/`fleet`, `render_user_data`, `code_sha`, `_provenance`) | Producer orchestration on object storage + spot fleets; lands with the build-phase milestone. |
| `prep_data.py`, `reader.py` → `TokenShard`, `verify.py`, `bench.py` | The Megatron/NeoX indexed-shard path; requires `megatron-core` at read time. Deferred as a unit. |
| `.github/` workflows | CI wiring is out of scope for milestone 1. |
| chunk prefetch (not in source either) | Deferred to milestone 3 — noted here and in the README so the absence is explicit. |
