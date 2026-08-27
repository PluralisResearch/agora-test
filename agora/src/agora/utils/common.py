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


import shutil

from pathlib import Path

from agora_server.hivemind.utils.logging import get_logger


logger = get_logger(__name__)


def infer_expert_params(
    pipeline_stage: str,
    uid: str,
) -> dict:
    """Infer required expert and stage parameters from pipeline_stage.

    Args:
        pipeline_stage (str): stage type in the format: head-X, body-X, tail-X (X is int)
        uid (str): expert UID assigned by the authorizer (e.g. head.0.42)

    Raises:
        ValueError: wrong stage type

    Returns:
        dict: expert_pattern, expert_name, stage
    """
    try:
        stage, stage_idx = pipeline_stage.split("-")
        stage_idx = int(stage_idx)
        assert stage in ["head", "body", "tail"]
    except Exception as e:
        raise ValueError("Wrong stage type. It should be one of: head-X, body-X, tail-X (X is int).") from e

    expert_name = f"lm_{stage}"

    params = {
        "expert_uids": [uid],
        "expert_name": expert_name,
        "stage": stage_idx,
    }

    return params


def clean_tmp(gpu_id: int = 0):
    """Remove tmp download files."""
    downloads_dir = Path(f"/tmp/agora_downloads_gpu{gpu_id}")
    if downloads_dir.exists():
        try:
            shutil.rmtree(downloads_dir)
            logger.info(f"{downloads_dir} folder was cleaned")
        except Exception as e:
            logger.warning(f"Can't remove {downloads_dir} folder: {e}")


def get_batch_size(gpu_memory: float, max_batch_size: int | dict[int, int]) -> int:
    """Select batch size based on GPU memory and max_batch_size config.

    Example of `max_batch_size` config:
    {
        8: 4,   # if 8 <= GPU VRAM < 16 GB, use batch size 4
        16: 8,  # if 16 <= GPU VRAM < 32 GB, use batch size 8
        32: 16, # if 32 <= GPU VRAM < 64 GB, use batch size 16
        64: 32, # if 64 <= GPU VRAM, use batch size 32
    }
    """
    if isinstance(max_batch_size, int):
        return max_batch_size

    # Find the largest threshold that fits within available VRAM
    eligible = {k: v for k, v in max_batch_size.items() if k <= gpu_memory}
    if not eligible:
        logger.error(f"GPU is not eligible (GPU VRAM: {gpu_memory} GB).")
        raise ValueError("GPU VRAM is too low.")

    best_tier = max(eligible)
    batch_size = eligible[best_tier]
    return batch_size
