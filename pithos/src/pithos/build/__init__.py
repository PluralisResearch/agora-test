"""Corpus-build (producer) side of pithos.

Milestone 1 ships the deterministic, dependency-free primitives the build
phases compose — keyed-hash ordering, tranche thresholds, record framing,
EOS-guarded tokenization, the overlap chunk writer, and the k-way merge —
re-exported here from `pithos.corpus`. They are pure Python/NumPy,
byte-identical for any work partition, and write the approved default
contract: flat little-endian int32 ('<i4') objects of 2**26 logical tokens
+ one overlap token (`pithos_stream_v1`).

Milestone 2 adds the deterministic build plane on top of those primitives:
`recipe` (schema + immutable lock phase with per-object content digests),
`sources` (pinned HF Parquet via the real list_repo_tree/hf_hub_download
APIs, local Parquet/JSONL, S3/R2, immutable HTTP, locked legacy-binary
streams — all over the `storage` abstraction with injectable
clients/fetchers), `policy` (anchored segment-aware selection, keyed order,
domain-separated validation reservation), `transforms` (a small versioned
recipe-controlled registry), `tokenize` (locked tokenizer policies with
asset/implementation evidence, exact [BOS]+tokens+[EOS] framing), `markers`
(part completion markers pinning every lock, byte-identical across worker
counts), `engine` (plan / worker / finalize / verify / publish —
bounded-memory, resumable, byte-identical across worker counts), and `cli`
(the `pithos` command). Build-only dependencies (pyarrow, boto3,
tokenizers, huggingface-hub) exist only under the `build` extra, so readers
never pay for them.
"""

from ..corpus import (
    ChunkWriter,
    key_of,
    merge_parts,
    numpy_sorted_records,
    read_records,
    t1_from_fraction,
    tokenize_docs,
    write_record,
)


__all__ = [
    "ChunkWriter",
    "key_of",
    "merge_parts",
    "numpy_sorted_records",
    "read_records",
    "t1_from_fraction",
    "tokenize_docs",
    "write_record",
]
