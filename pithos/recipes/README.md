# Recipes

A recipe is the pinned definition of a corpus: format contract, dataset +
revision, tokenizer + revision, hash seed, key size, chunk geometry,
selection policy, include/exclude rules, validation reservation, cross-crawl
duplicate policy, and the planned outputs. **The pins ARE the corpus
identity** — a built corpus is identified by its `manifest.json` canonical
sha256 (verified by pithos at read time), and any edit to a recipe's pins
defines a NEW corpus, deliberately, never as a drive-by.

Recipes are data (JSON), not code: the build phases (later milestones)
consume them; the reader never does (a built corpus is self-describing via
its manifest).

## Index

| file | kind | status |
|---|---|---|
| `fineweb_edu_pithos_v1.json` | recipe | **planned — not built.** The approved `pithos_stream_v1` contract: flat `<i4` objects, 2^26 logical tokens + 1 overlap, seq = powers of two 2K–128K. Builds the full per-crawl FineWeb-Edu corpus (samples excluded) plus the published 0.1% validation split. Lock-phase pins are explicit `null` until lock time. |
| `fineweb_edu_sample_350bt_pithos_v1.json` | recipe | **planned — not built.** The standalone `sample/350BT` reproduction, built only from its own source objects — never a key-order prefix of the per-crawl corpus. |
| `legacy_fineweb_edu_rechunk_v1.json` | recipe | **planned — not built.** Byte-preserving re-chunk of a locked legacy flat-token stream into the approved contract. Its source uri, ordered inventory, and expected logical token count are REQUIRED lock inputs (pass `--source-uri` / `--publish-root` at lock time). |
| `legacy_fineweb_edu_tranche1.snapshot.json` | provenance snapshot | Records the EXTERNALLY built (pretrain-data/colonnade) legacy tranche's pins so readers can consume that corpus and its pins are never silently reused. Pithos did not build it. |
