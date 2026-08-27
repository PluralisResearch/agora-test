"""Runtime cache-plane configuration as one immutable, validated value.

Validation runs at construction, so a bad value fails at config time
rather than inside a worker.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cache import DEFAULT_BUDGET_BYTES
from .errors import CacheError
from .prefetch import DEFAULT_PREFETCH_DEPTH


@dataclass(frozen=True)
class CacheConfig:
    """Immutable runtime cache configuration, validated at construction.

    Args:
        cache_dir: directory holding leased chunk objects.
        budget_bytes: aggregate byte budget for managed on-disk objects
            across processes sharing `cache_dir`; a positive int, default
            1 GiB. Processes using different cache directories have
            independent budgets.
        prefetch_depth: maximum prefetch requests in flight; a nonnegative
            int (0 disables prefetching), default 2.

    Raises:
        CacheError: if `budget_bytes` is not a positive int or
            `prefetch_depth` is not a nonnegative int.
    """

    cache_dir: str
    budget_bytes: int = DEFAULT_BUDGET_BYTES
    prefetch_depth: int = DEFAULT_PREFETCH_DEPTH

    def __post_init__(self) -> None:
        if isinstance(self.budget_bytes, bool) or not isinstance(self.budget_bytes, int):
            raise CacheError(f"budget_bytes must be a positive int, got {self.budget_bytes!r}")
        if self.budget_bytes < 1:
            raise CacheError(f"budget_bytes {self.budget_bytes} < 1")
        if isinstance(self.prefetch_depth, bool) or not isinstance(self.prefetch_depth, int):
            raise CacheError(f"prefetch_depth must be a nonnegative int, got {self.prefetch_depth!r}")
        if self.prefetch_depth < 0:
            raise CacheError(f"prefetch_depth {self.prefetch_depth} < 0")
