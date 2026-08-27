"""Shared helpers for model compression artifacts."""

import torch
import torch.nn.functional as F

from agora_server.hivemind.utils import get_logger


logger = get_logger(__name__)


def _fit_fixed_token_weight(
    weight: torch.Tensor,
    expected_shape: torch.Size,
    allow_padding: bool,
) -> torch.Tensor:
    if weight.shape == expected_shape:
        return weight
    can_pad = (
        allow_padding
        and weight.ndim == 2
        and len(expected_shape) == 2
        and weight.shape[0] < expected_shape[0]
        and weight.shape[1] == expected_shape[1]
    )
    if not can_pad:
        raise ValueError(f"fixed token weight shape {tuple(weight.shape)} does not match {tuple(expected_shape)}")
    logger.warning(
        f"Zero-padding frozen compression embeddings from {weight.shape[0]} to {expected_shape[0]} vocab rows"
    )
    return F.pad(weight, (0, 0, 0, expected_shape[0] - weight.shape[0]))
