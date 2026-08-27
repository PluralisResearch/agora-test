# Copyright 2026 Pluralis Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

import torch

from pydantic import BaseModel, computed_field

from agora_server.hivemind.utils import get_logger
from agora_server.models.base_arguments import ModelArguments, TransformerArgumentsBase
from agora_server.models.reparam_llama.components import ReparametrizedEmbedding


logger = get_logger(__name__)


# GPU peak FLOPS specifications per precision. All values are DENSE tensor-core
# throughput at boost clock (no 2:4 sparsity). Sources: NVIDIA architecture
# whitepapers and per-GPU datasheets.
SUPPORTED_PRECISIONS = ("fp32", "tf32", "bf16")
# fmt: off
GPU_PEAK_FLOPS = {
    # Ampere — data-center / cloud
    "A100":         {"fp32": 19.5e12,  "tf32": 156e12,    "bf16": 312e12},      # https://www.nvidia.com/en-us/data-center/a100/ — 40GB and 80GB have identical FLOPS
    "A40":          {"fp32": 37.4e12,  "tf32": 74.8e12,   "bf16": 149.7e12},    # https://images.nvidia.com/content/Solutions/data-center/a40/nvidia-a40-datasheet.pdf
    "A10G":         {"fp32": 31.5e12,  "tf32": 62.5e12,   "bf16": 125e12},      # AWS-only variant of A10
    "A10":          {"fp32": 31.2e12,  "tf32": 62.5e12,   "bf16": 125e12},      # https://www.nvidia.com/ja-jp/data-center/products/a10-gpu/
    # Hopper — data center
    "H100-PCIe":    {"fp32": 51e12,    "tf32": 378e12,    "bf16": 756e12},      # https://www.nvidia.com/en-us/data-center/h100/
    "H100-NVL":     {"fp32": 60e12,    "tf32": 418e12,    "bf16": 836e12},      # https://www.nvidia.com/en-us/data-center/h100/
    "H100":         {"fp32": 67e12,    "tf32": 495e12,    "bf16": 989e12},      # SXM5 — default H100 catch-all
    "H200":         {"fp32": 67e12,    "tf32": 495e12,    "bf16": 989e12},      # https://www.nvidia.com/en-us/data-center/h200/
    # Turing — no TF32
    "T4":           {"fp32": 8.1e12,   "tf32": 8.1e12,    "bf16": 65e12},
    # Ada Lovelace — data center
    "L4":           {"fp32": 30.3e12,  "tf32": 60e12,     "bf16": 121e12},      # NVIDIA L4 datasheet quotes 120/242 with sparsity; dense is half
    "L40S":         {"fp32": 91.6e12,  "tf32": 183e12,    "bf16": 362e12},      # https://www.nvidia.com/en-in/data-center/l40s/
    "L40":          {"fp32": 90.5e12,  "tf32": 90.5e12,   "bf16": 181e12},      # consumer-style tensor config (TF32 = FP32)
    # Blackwell — data center
    "B200":         {"fp32": 75e12,    "tf32": 1125e12,   "bf16": 2250e12},     # https://www.nvidia.com/en-us/data-center/hgx/ — per-GPU dense
    "B300":         {"fp32": 75e12,    "tf32": 1125e12,   "bf16": 2250e12},     # Blackwell Ultra — same compute as B200; differs in memory
    # Workstation / Pro — Ampere
    "RTX A6000":    {"fp32": 38.7e12,  "tf32": 77.4e12,   "bf16": 155e12},
    "RTX A5000":    {"fp32": 27.8e12,  "tf32": 55.5e12,   "bf16": 111e12},
    "RTX A4000":    {"fp32": 19.2e12,  "tf32": 38.3e12,   "bf16": 76.7e12},
    # Workstation / Pro — Ada / Blackwell
    "RTX 6000 Ada": {"fp32": 91.1e12,  "tf32": 182e12,    "bf16": 364e12},      # https://www.nvidia.com/en-us/products/workstations/rtx-6000/
    "RTX PRO 6000": {"fp32": 126e12,   "tf32": 252e12,    "bf16": 504e12},      # https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/
    # GeForce — Ampere consumer (TF32 = 2x FP32, BF16 = 4x FP32 dense)
    "RTX 3090":     {"fp32": 35.6e12,  "tf32": 71.15e12,  "bf16": 142.3e12},
    "RTX 3090 Ti":  {"fp32": 40e12,    "tf32": 80e12,     "bf16": 160e12},      # NVIDIA marketing page quotes 160/320 with sparsity; dense is half
    # GeForce — Ada/Blackwell consumer (TF32 = FP32, no tensor TF32 accel; BF16 = 2x FP32 dense)
    "RTX 4090":     {"fp32": 82.6e12,  "tf32": 82.6e12,   "bf16": 165.16e12},   # Ada whitepaper Table 2: 82.58 / 165.16 dense
    "RTX 5090":     {"fp32": 105e12,   "tf32": 105e12,    "bf16": 209.5e12},    # Blackwell whitepaper: 105 / 209.5 dense
    "RTX 5080":     {"fp32": 56.3e12,  "tf32": 56.3e12,   "bf16": 112.6e12},
    "RTX 5070 Ti":  {"fp32": 43.9e12,  "tf32": 43.9e12,   "bf16": 87.8e12},
    "RTX 5070":     {"fp32": 30.9e12,  "tf32": 30.9e12,   "bf16": 61.8e12},
    "RTX 5060 Ti":  {"fp32": 23.7e12,  "tf32": 23.7e12,   "bf16": 47.4e12},
    # Grace Blackwell — DGX Spark
    "GB10":         {"fp32": 18.5e12,  "tf32": 18.5e12,    "bf16": 96.9e12},     # DGX Spark / GB10 from public benchmark.
}
# fmt: on


def _normalize_gpu_name(name: str) -> str:
    """Normalize GPU names for case-insensitive, punctuation-tolerant matching."""
    return " ".join(re.sub(r"[^0-9a-z]+", " ", name.casefold()).split())


GPU_PEAK_FLOPS_KEYS = sorted(
    ((_normalize_gpu_name(gpu_key), gpu_key) for gpu_key in GPU_PEAK_FLOPS),
    key=lambda item: len(item[0]),
    reverse=True,
)


class NumFlopPerSample(BaseModel):
    """Number of floating point operations per sample for forward and backward passes."""

    forward: int
    backward: int

    @computed_field
    @property
    def total(self) -> int:
        return self.forward + self.backward


def get_num_flop_per_sample(
    num_params: int,
    model_config: ModelArguments,
) -> NumFlopPerSample | None:
    """Compute number of floating point operations per sample based on model type."""
    if isinstance(model_config, TransformerArgumentsBase):
        return get_num_flop_per_sample_transformer(num_params, model_config)
    else:
        logger.warning(f"FLOP computation not implemented for model type: {type(model_config)}")
        return None


def get_num_flop_per_sample_transformer(
    num_params: int,
    model_config: TransformerArgumentsBase,
) -> NumFlopPerSample:
    """Compute number of floating point operations per sample for transformer models."""
    layers = model_config.n_layers
    heads = model_config.n_heads
    head_dim = model_config.hidden_dim // model_config.n_heads
    seq_len = model_config.max_seq_len

    flop_per_token_fwd = 2 * num_params + 4 * layers * heads * head_dim * seq_len
    flop_per_token_bwd = 4 * num_params + 8 * layers * heads * head_dim * seq_len

    return NumFlopPerSample(
        forward=flop_per_token_fwd * seq_len,
        backward=flop_per_token_bwd * seq_len,
    )


def get_num_params(model: torch.nn.Module, exclude_embedding: bool = False) -> int:
    """Compute number of model parameters."""
    num_params = sum(p.numel() for p in model.parameters())

    if exclude_embedding:
        for module in model.modules():
            if isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag, ReparametrizedEmbedding)):
                num_params -= sum(p.numel() for p in module.parameters())

    return num_params


def get_peak_flops(precision: str) -> float | None:
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(f"Unsupported precision {precision!r}, expected one of {SUPPORTED_PRECISIONS}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("No cuda device detected. Cannot get device peak flops.")
        return None

    device_info = torch.cuda.get_device_properties(device)
    device_name = device_info.name
    logger.info(f"Identified device is: {device_name}")
    normalized_device_name = _normalize_gpu_name(device_name)

    # Find matching GPU architecture
    for normalized_gpu_key, gpu_key in GPU_PEAK_FLOPS_KEYS:
        if normalized_gpu_key in normalized_device_name:
            peak_flops = GPU_PEAK_FLOPS[gpu_key][precision]
            logger.info(f"Peak flops for {gpu_key} at {precision}: {peak_flops:.3e}")
            return peak_flops

    # None for unknown devices
    logger.warning(f"Peak flops undefined for: {device_name}")
    return None
