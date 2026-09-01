# pithos

Deterministic pretraining data for swarm training. One contract underlies
everything here: **`sample_id → tokens` is a pure function** — the same id
yields the same tokens on any machine, any number of workers, forever.

**Status: standalone data plane implemented** — deterministic corpus builds,
immutable publication, verified loading from local storage, HTTP/R2, and S3,
the hardened streaming cache, and the public NumPy/Torch read APIs are in
place. Trainer integration is intentionally a separate change; production
corpus builds and performance canaries remain operational gates.

## The approved default contract: `pithos_stream_v1`

- Flat **little-endian int32 (`<i4`)** chunk objects of **2²⁶ logical tokens
  (256 MiB) + one overlap token**, so any `seq+1` sample lives inside exactly
  one object. Bytes carry no seq, no shuffle, no mask metadata.
- Sequence length is a **read-time** parameter from the supported set:
  powers of two from **2K (2048) through 128K (131072)**.
- Every chunk carries a **sha256** that is verified before any byte is
  mapped; corpus identity is the manifest's **canonical sha256**, computed
  over the manifest content and checked — never accepted on trust.
- The legacy `colonnade_stream_v1` format (2³⁰-token chunks) is readable
  through explicit compatibility paths only — it is not the new end state.

## Install

pithos is installed from a local checkout of this repository — it is not
published to an external package index.

```bash
pip install ./pithos            # base: Python + NumPy only
pip install ./pithos[s3]        # + reader-side s3:// chunk access (boto3)
pip install ./pithos[build]     # + corpus-build deps (pyarrow, boto3, tokenizers, ...)
pip install ./pithos[test]      # + pytest
```

**PyTorch is never installed or selected by pithos.** The base runtime is
Python + NumPy only; `pithos.torch` imports the torch already
present in your environment (training images have one) and refuses with a
clear error otherwise. There is deliberately no `torch` extra.

## What milestone 1 provides

- `pithos.corpus` / `pithos.build` — the deterministic
  build primitives: keyed-hash ordering, tranche thresholds, record framing,
  EOS-guarded tokenization, the overlap chunk writer, and the k-way merge.
  Byte-identical output for any work partition; typed errors, never
  assert/SystemExit.
- `pithos.manifest` — manifest validation (structure, safe chunk
  names, per-chunk sha256, canonical identity, the supported seq set) before
  anything is created or downloaded.
- `pithos.cache` — verified, byte-budgeted (1 GiB default) LRU
  chunk leasing: atomic publication, per-chunk interprocess locks, no
  startup deletion, explicit `<i4` mmaps.
- `pithos.torch.StreamShard` — the **reference reader**, kept
  behavior-identical to the imported golden vectors.

## What milestone 2 provides: the build plane

`pithos.build` turns a recipe into an immutable, verified,
published corpus — deterministically:

```bash
pithos recipe lock RECIPE [--publish-root U] [--source-uri U]
                                           # resolve+pin source/tokenizer/code revisions and the per-corpus seed
pithos build plan RECIPE [--build-root U] # scan inventory, select work units, publish plan.json
pithos build worker URI --worker-index N --worker-count M
pithos build finalize URI                 # keyed global merge → strict-geometry chunks + manifests
pithos build verify URI                   # independent re-verification + reconciliation
pithos publish URI                        # create-once immutable publication
```

- **Deterministic by construction.** Work units are the selected source
  items in locked (sorted) order; worker N of M claims positions ≡ N (mod M).
  Assignment changes only which part file a record lands in — never the
  final bytes — because finalize merges ALL parts in global keyed
  BLAKE2b-128(crawl, document_id) order and chunking uses fixed stream positions
  (strictly 2²⁶+1 for `pithos_stream_v1`). Plan, parts, markers, chunks,
  manifests, finalize and verify receipts are byte-identical across worker
  counts and retry assignments — markers pin no worker identity.
- **Bounded memory at every phase.** The plan scan is metadata-only:
  Parquet row evidence comes from two small ranged reads per object
  (footer tail + footer), pinned into create-once resumable `scan/` state
  — never a full download. Workers open ONE source object at a time (local
  files hashed from the very descriptor that is then parsed — no
  hash-then-reopen window; remote objects streamed to a collision-free,
  content-named local spool that is verified and reclaimed before the next
  unit), Parquet is iterated in row batches, tokenization runs in bounded
  batches, each unit's records are block-sorted and spilled to local run
  files, and every merge (runs per unit, parts at finalize) is a bounded
  fan-in iterative streaming k-way merge whose intermediate runs are
  reclaimed on success and on failure. Finalize emits one chunk at a time
  straight to storage; all hashing is incremental. Nothing holds a
  corpus-sized or object-set-sized list of bytes.
- **Resumable.** Every completed unit publishes a completion marker pinning
  the recipe/source/tokenizer locks, the code revision and pithos
  build-code digest, the plan unit's exact source-item identity
  (name/size/sha256), document/token counts, and part sha256s under the
  unit's own canonical part names. The build-code digest covers the build
  package and build-affecting root modules, with framed path and content
  bytes; runtime-only reader edits do not require a source relock. The digest
  is not merely pinned: every engine phase (plan, worker, finalize, verify,
  publish) first recomputes it and refuses to operate under code different
  from the locked build code. A valid marker skips the unit; a marker
  repointed at another item or another unit's checksum-valid part, a
  role/count invariant violation, or a corrupt artifact is rejected, never
  silently rebuilt over. Publication is atomic create-once at every level
  (plan, parts, markers, chunks, manifests, published outputs): a name
  that already exists is verified byte-identical (idempotent retry —
  including mid-phase crashes, concurrent duplicate workers, and
  interrupted publishes, whose surviving chunks are hash-verified and
  skipped) or rejected loudly — never silently overwritten.
- **Sources.** Pinned Hugging Face Parquet via the real
  `list_repo_tree`/`HfFileSystem` APIs (the lock pins the exact commit and
  every selected file's size + LFS `BlobLfsInfo.sha256`; workers stream
  pinned-revision objects through ranged reads into the verified spool —
  the hub cache is never used for source data), local Parquet/JSONL,
  S3/R2 (ranged GETs for footer evidence; conditional-put publication;
  `?endpoint_url=...&profile=...` URI params carry non-secret endpoint
  config; S3-compatible backends must honor `If-None-Match: *` for
  create-once publication), explicitly declared immutable-HTTP items
  (byte-counted and digest-pinned at lock time through the CLI; object names are
  percent-encoded per path segment so reserved characters can never change
  the addressed resource), and locked legacy-binary
  streams for byte-preserving re-chunking. Every source object carries
  content sha256 evidence in the lock, verified at read time. Tests use
  protocol-faithful fakes — no credentials or network.
- **Policy.** Recipe-controlled, versioned: anchored segment-aware
  include/exclude selection (`*` never spans `/`, so
  `data/CC-MAIN-*/*.parquet` is exact), the 0.1% domain-separated
  validation reservation published as its own output, exact
  `[BOS] + tokens + [EOS]` framing under the locked tokenizer (asset digest
  + implementation version pinned and verified), per-crawl count
  reconciliation against real footer row evidence, a small versioned
  transform registry (FineWeb-Edu: none — upstream only), and named outputs
  with optional exact token budgets (an N-token output holds exactly N
  logical tokens, cut mid-document at the boundary). FineWeb-Edu:
  cross-crawl duplicates retained, never global-deduped, and the legacy dev
  repetition filter is never applied (no such transform exists).
  Source text that tokenizes to an interior BOS token retains it as content;
  only EOS is reserved as the flat-stream document terminator and removed
  from document bodies before the final EOS is appended.
- **Variants are truthful.** The full per-crawl FineWeb-Edu corpus and its
  published 0.1% validation split are one recipe
  (`fineweb_edu_pithos_v1.json`); the standalone `sample/350BT`
  reproduction is a separate recipe built ONLY from its own source objects
  (`fineweb_edu_sample_350bt_pithos_v1.json`); a legacy flat-token stream
  is re-chunked byte-preservingly into 2²⁶+1 objects by a generic recipe
  (`legacy_fineweb_edu_rechunk_v1.json`, uri and expected logical token
  count null = REQUIRED lock inputs; finalize reconciles the emitted total
  against the locked inventory). None is a key-order prefix of another.
- **Verification.** `verify` re-validates every manifest, re-hashes every
  chunk (streamed), reconciles marker counts against manifests and declared
  per-crawl source counts, and proves train/validation split separation
  from the parts themselves.

This subsumes the legacy `datasets/scripts/fineweb_shuffle_reshard`
pipeline: the keyed global order replaces the approximate/full per-dump
shuffles and the producer/consumer merge-reshard (no trainer-count-dependent
shard ranges, no imbalance), and marker/manifest reconciliation replaces the
doc-count verification scripts.

## What phase 3 provides: the public runtime

The approved public read API, implemented on top of the hardened cache:

- `pithos.config.CacheConfig` — the immutable runtime cache
  configuration, validated at construction: `cache_dir`, a `budget_bytes`
  aggregate byte budget for managed on-disk objects across every process
  sharing that directory (default **1 GiB**), and `prefetch_depth` (default
  **2**; **0 disables** prefetching). Separate cache directories have
  independent budgets; resident mmap accounting is per process.
- `pithos.runtime.Corpus` — a deterministic sample-id-addressed
  corpus handle, NumPy-only. It binds a **local, verified manifest**
  (structure, per-chunk sha256, and the computed canonical identity —
  optionally pinned via `expected_manifest_identity`) and its sample layout
  to the chunk cache: everything that can fail validation fails in the
  constructor, before a byte is fetched. `sequence_length` is a strict
  read-time parameter: `pithos_stream_v1` supports powers of two from
  2K–128K, while legacy manifests retain their declared divisible lengths.
  `read(start, rows)` takes an **absolute sample index** (wrapping by
  modulo `total_samples`, advancing the epoch) and returns a NumPy
  **int32 `[rows, seq+1]`** array — each sample plus its overlap token.
  `read_strided(start, stride, rows)` reads absolute indices
  `start, start + stride, ...`, which lets stream `d` of `N` consume
  `d, d + N, d + 2N, ...` without trainer-count-dependent objects.
  `identity` and `total_samples` expose the canonical corpus identity and
  the samples per epoch. Context-managed; `close` shuts down prefetch and
  releases this process's mmaps, never touching on-disk objects.
- `Corpus.from_uri(...)` — the publication-facing loader. Give it a local,
  `file://`, HTTP(S), or `s3://` publication directory (or its
  `manifest.json` URI); it fetches and validates the bounded manifest,
  optionally checks its immutable identity pin, derives the complete chunk
  locator set, and leaves object transfer lazy. Pinned remote manifests are
  cached under the runtime cache and can be reopened without another
  manifest request. `s3://` URIs may carry non-secret endpoint parameters —
  `?endpoint_url=...&region=...&profile=...` — so S3-compatible services
  (private R2) work first-class; the parameters propagate to every derived
  chunk locator, credentials stay in the environment or the named profile,
  and each transfer signs from its own fresh boto3 session (the process-wide
  default session's cached credentials are never trusted).
- `Corpus.from_name(...)` — the registry loader. `pithos/corpora.yaml` maps
  corpus NAMES to non-secret publication facts (URI with endpoint
  parameters, the immutable `manifest_identity` pin, per-corpus reader cache
  overrides), so a consumer passes a name — e.g. `fineweb_edu_1T` — and
  the reader figures the rest out. YAML
  parsing uses the consumer's own PyYAML (as `pithos.torch` uses the
  consumer's torch); see `pithos.registry`.
- **Leasing, transport, and prefetch.** A non-empty `chunk_locators`
  mapping puts the cache in leasing mode: chunks are fetched lazily under
  per-chunk interprocess file locks, verified against the manifest sha256
  before publication, published atomically (same-directory rename), and
  bounded on disk by the byte-budgeted LRU. Downloads are **resumable**:
  progress is journaled to a sidecar keyed on the manifest-object identity
  (never the locator), and a later attempt re-hashes the vouched prefix,
  then resumes with Range/If-Range — over plain local paths (seek plus an
  inode/size/mtime change stamp) and file/HTTP/HTTPS URLs in the base, and
  over `s3://bucket/key` via a lazily imported boto3 (the `s3` extra).
  Prefetch is a bounded, single-threaded pipeline through the cache's own
  fetch path, so warmed bytes pass the same verification, locking, and
  budget enforcement as an on-demand read.
  Eviction transactions serialize briefly under one cache-directory lock and
  may wait for a process currently using a selected victim's chunk lock; size
  a shared cache for the aggregate active working set to avoid churn. A newly
  published protected object can exceed the limit only until enforcement
  completes, and a single object larger than the configured limit is retained
  as the documented oversized-object exception.
- `pithos.torch.PithosBatchSource` — the public training-facing
  batch source. It **owns** one Corpus (closing the source closes the
  corpus) and serves the same absolute-range or strided reads as CPU
  **torch.int64 `[rows, seq+1]`** tensors: one matching Corpus call per
  read, and only the selected int32 batch is widened to int64 — corpus
  objects and cached mmaps are never cast.

`StreamShard` remains the reference reader (see the legacy note below); it
is not the final interface.

## Remaining operational work

- **Authentication-backed leases** — trainerless W2W currently assigns
  deterministic `data_idx` strides. Run-scoped lease allocation is a later
  coordination change; it does not change the corpus format or read API.
- **Live production FineWeb-Edu runs and performance canaries** — the
  Hugging Face, S3/R2, and HTTP engine paths are implemented against the
  real API signatures and covered by protocol-faithful fakes, but have NOT
  yet been run against the production FineWeb-Edu corpus (which needs no
  credentials for the public repo, but does need network; per-worker spool
  demand is one source object at a time, ~GBs, not the corpus), and no
  training performance canaries have run.
- The legacy re-chunk additionally needs the real locked legacy stream
  location — the recipe's uri is null ON PURPOSE and locking fails clearly
  until it is supplied.

## Reading a corpus: the public runtime

```python
from pithos import CacheConfig, Corpus
from pithos.torch import PithosBatchSource  # imports YOUR torch

config = CacheConfig(cache_dir="corpus/")  # defaults: 1 GiB on-disk budget, prefetch depth 2

with (
    Corpus.from_uri(
        "https://data.example.org/fineweb-edu/full",
        sequence_length=4096,  # strict: a supported power of two, 2K..128K
        seed=YOUR_SEED,
        cache_config=config,
        expected_manifest_identity="<canonical manifest sha256>",  # optional pin
    ) as corpus,
    PithosBatchSource(corpus) as source,
):
    batch = source.read(start=0, rows=8)  # CPU torch.int64 [8, seq+1]
```

The URI may instead be a local publication directory, its exact
`manifest.json`, a `file://` URI, or `s3://bucket/prefix` (optionally with
`?endpoint_url=...&region=...&profile=...` for S3-compatible services).
Public R2 is read over HTTPS without an SDK; private R2/S3 needs the `s3`
extra. A registry name does the same with one line — for entries that carry
a `uri` (the packaged `fineweb_edu_1T` entry deliberately does not: its
location arrives at run time and the entry only pins the identity):

```python
corpus = Corpus.from_name("my_corpus_with_a_uri", 2048, YOUR_SEED, "corpus/")
```

2**30-chunk corpora carry reader overrides in their registry entries on
purpose: one chunk is 4 GiB, so the library cache defaults (1 GiB budget,
prefetch depth 2 — sized for the approved 2**26 contract, where three chunks
fit the budget) would make the prefetched neighbour and the current chunk
evict each other in a re-download loop. The rule when configuring by hand:
`budget_bytes >= chunk_bytes * (prefetch_depth + 1)`. For the lower-level
constructor, pass an explicit chunk-name-to-locator mapping covering the
manifest exactly, or `{}` when the manifest and all objects are already in
`cache_dir`.

`read(start, rows)` uses **absolute sample-range semantics** matching lease
allocation: `start` is an absolute index into the epoch stream, wrapping by
modulo `total_samples`; `seed` is yours. Locators carry object locations,
never secrets. The base runtime reads plain local paths and file/HTTP/HTTPS
URLs — so a public R2 bucket is read over plain HTTPS with no extra and no
credentials — while `s3://` needs the `s3` extra. NumPy-only consumers skip
torch entirely and call `corpus.read(start, rows)` for an int32 array.

### Legacy reference: `StreamShard`

```python
from pithos.torch import StreamShard

# a pre-placed local corpus: manifest.json + chunk objects in one directory
shard = StreamShard("corpus/manifest.json", cache_dir="corpus/", chunk_urls={}, seq=4096, seed=YOUR_SEED)
batch = shard.ids(batch_id=0, rows=8)  # torch.LongTensor [8, seq+1]
```

**`ids(batch_id, rows)` uses legacy batch-index semantics**: it returns
samples `[batch_id*rows, batch_id*rows + rows)` — `batch_id` is NOT an
absolute sample-range start and does not express lease allocation. Feed
`batch_id = step` and resume is "start from the step number". Leasing mode
(`chunk_urls` given) fetches chunks lazily under the on-disk byte budget
and verifies every one.

The legacy reader contract is pinned by golden vectors
(`tests/test_stream_reader.py`): a port is correct iff it reproduces them. A
digest change is a NEW contract and ships as a new manifest format tag, never
as a silent edit.

## Layout

```
pithos/
├── pyproject.toml            # dist: pluralis-pithos (base deps: numpy only)
├── README.md
├── PROVENANCE.md             # what was imported from pretrain-data, and how
├── recipes/                  # corpus definitions + provenance snapshots (pins ARE identity)
├── src/pithos/
│   ├── __init__.py           # version + public re-exports (primitives, CacheConfig, Corpus)
│   ├── errors.py             # PithosError hierarchy (typed ValueErrors)
│   ├── manifest.py           # pithos_stream_v1 + legacy: validation, per-seq layout, identity
│   ├── corpus.py             # deterministic build primitives (pure Python/NumPy)
│   ├── config.py             # CacheConfig: cache dir, byte budget, prefetch depth
│   ├── cache.py              # verified chunk leasing: locks, atomic publish, resumable fetch, byte-budget LRU
│   ├── transport.py          # local / file / HTTP(S) / s3 byte transport, journaled Range resume
│   ├── prefetch.py           # bounded single-threaded prefetch through the cache's fetch path
│   ├── publication.py        # publication URI -> verified local manifest + lazy chunk locators
│   ├── runtime.py            # Corpus: the public NumPy reader (absolute reads, int32 [rows, seq+1])
│   ├── torch.py              # StreamShard reference reader + PithosBatchSource (imports YOUR torch; optional)
│   └── build/                # the deterministic build plane + CLI (build extra deps)
│       ├── recipe.py         # recipe schema + the immutable lock phase
│       ├── storage.py        # local / S3-R2 / immutable-HTTP storage, atomic create-once
│       ├── sources.py        # source adapters: locked inventory, verified spool, document iteration
│       ├── policy.py         # segment-aware selection, keyed order, validation reservation
│       ├── transforms.py     # the versioned transform registry (FineWeb: none)
│       ├── tokenize.py       # locked tokenizer policies, exact [BOS]+tokens+[EOS]
│       ├── markers.py        # part completion markers (pin every lock; no worker identity)
│       ├── engine.py         # bounded-memory plan / worker / finalize / verify / publish
│       └── cli.py            # the `pithos` command
└── tests/
```

## Tests

```bash
pip install -e ./pithos[test]
pytest pithos/tests
```

Without torch, the base suite skips only the torch-dependent tests — the
PithosBatchSource adapter tests (`test_torch_adapter.py`), the
golden-vector StreamShard reader tests (`test_stream_reader.py`), and the
Corpus/torch parity pins — cleanly; a training or system environment that
already has torch runs them. Everything else runs on Python + NumPy alone
(build-engine tests needing pyarrow/tokenizers skip without the `build`
extra).

## Provenance

The deterministic primitives and the reference reader were imported from
[pretrain-data](https://github.com/PluralisResearch/pretrain-data) and remain
behavior-identical to the readers its trainers consume, with milestone-1
review hardening on top (typed errors, verified integrity, path containment,
interprocess locks, byte budget). See [PROVENANCE.md](PROVENANCE.md) for the
exact source commit, the file-by-file mapping, every deliberate divergence,
and what was deliberately not imported yet.
