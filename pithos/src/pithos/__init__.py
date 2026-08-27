"""pithos — deterministic pretraining data pipelines.

One contract: `sample_id -> tokens` is a pure function — the same id yields
the same tokens on any machine, any number of workers, forever.

Base runtime is Python + NumPy only. The torch-returning reader lives in
`pithos.torch` (imported explicitly, using the consumer's own
torch — pithos never requires or selects PyTorch). Corpus-build phases that
need heavy deps live under `pithos.build` (the `build` extra).
"""

from .config import CacheConfig
from .corpus import (
    ChunkWriter,
    key_of,
    merge_parts,
    numpy_sorted_records,
    read_records,
    t1_from_fraction,
    tokenize_docs,
    write_record,
)
from .errors import CacheError, CorpusError, ManifestError, PithosError, RegistryError
from .manifest import (
    LEGACY_STREAM_V1,
    PITHOS_STREAM_V1,
    SUPPORTED_SEQ,
    ChunkInfo,
    Manifest,
    StreamLayout,
    manifest_sha256,
    validate_chunk_name,
)
from .runtime import Corpus


__version__ = "0.1.0"

__all__ = [
    "PITHOS_STREAM_V1",
    "LEGACY_STREAM_V1",
    "SUPPORTED_SEQ",
    "CacheConfig",
    "CacheError",
    "ChunkInfo",
    "ChunkWriter",
    "PithosError",
    "RegistryError",
    "Corpus",
    "CorpusError",
    "Manifest",
    "ManifestError",
    "StreamLayout",
    "key_of",
    "manifest_sha256",
    "merge_parts",
    "numpy_sorted_records",
    "read_records",
    "t1_from_fraction",
    "tokenize_docs",
    "validate_chunk_name",
    "write_record",
]
