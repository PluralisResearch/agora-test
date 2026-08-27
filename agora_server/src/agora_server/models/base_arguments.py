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

from abc import ABC
from typing import Any

from pydantic import BaseModel, computed_field


class ModelArguments(BaseModel):
    """Base class for model arguments."""

    stage: int | None = None  # Stage in the pipeline
    expert_name: str | None = None  # Name of the expert type used in the model

    # Compression parameters
    use_compression: bool = False  # Whether to use compression
    compression_rate: int = 100  # Higher means more compression
    ss_component_path: str | None = None  # Location of subspace components rcv, fixed_tok_weights

    @computed_field
    @property
    def input_schema(self) -> dict[str, Any]:
        """Returns a dictionary of parameters needed to generate sample input.

        This property must be overridden in subclasses to define model-specific input schemas.
        The schema is used to dynamically build sample inputs.

        The returned dictionary should contain dimension parameters that describe the shape and
        structure of the model's expected input tensors (omitting batch dimension).

        Examples:
            For a transformer head (LlamaArguments with expert_name="lm_head"):
            >>> return {"sequence_length": 512}

            For a transformer body with compression (LlamaArguments with use_compression=True):
            >>> compression_length = int(self.hidden_dim // self.compression_rate)
            >>> hid_dim = compression_length + 1
            >>> return {"sequence_length": 2048, "hid_dim": hid_dim}

            Usage:
            >>> model_args = ModelArgumentsRegistry.get("llama", "debugmodel_C")
            >>> sample_input_func = ExpertRegistry.get_sample_input(model="llama", expert_name="lm_head")
            >>> input_schema = model_args.input_schema
            >>> sample_input = sample_input_func(DUMMY_BATCH_SIZE, **input_schema)

        Returns:
            dict[str, Any]: A dictionary mapping parameter names to their values,
                describing the input tensor dimensions for the model (omitting batch dimension).
        """
        raise NotImplementedError("Subclasses must implement input_schema")


class TransformerArgumentsBase(ModelArguments, ABC):
    """Base class for transformer-based model arguments."""

    hidden_dim: int
    n_heads: int
    n_layers: int
    max_seq_len: int
    pad_fixed_token_embeddings: bool = True
