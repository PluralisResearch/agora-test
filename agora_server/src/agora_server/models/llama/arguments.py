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

from typing import Any

from pydantic import Field, computed_field

from agora_server.models.arguments_registry import ModelArgumentsRegistry
from agora_server.models.base_arguments import TransformerArgumentsBase


class LlamaArguments(TransformerArgumentsBase):
    model_config = {"validate_assignment": True}

    hidden_dim: int = 4096
    n_heads: int = 32
    n_layers: int = 32  # Total number of layers in the model, only used if replace_n_layers is False
    num_hidden_layers: int = 4  # This is the number of layers in a stage. Overwrites n_layers
    # If set, overwrites num_hidden_layers for the head stage
    num_hidden_layers_head: int | None = Field(frozen=True, default=None)
    # If set, overwrites num_hidden_layers for the body stage
    num_hidden_layers_body: int | None = Field(frozen=True, default=None)
    # If set, overwrites num_hidden_layers for the tail stage
    num_hidden_layers_tail: int | None = Field(frozen=True, default=None)
    # If True, n_layers is replaced by num_hidden_layers IMPORTANT: this should be set before expert_name is set!
    replace_n_layers: bool = True
    n_kv_heads: int | None = None
    vocab_size: int = 50265  # Using AutoTokenizer.from_pretrained("facebook/opt-2.7b") to match default
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    max_seq_len: int = 512

    # If `True`, then each transformer block init uses its layer ID, and if `False`, each uses the total number of transformer blocks
    depth_init: bool = True
    constant_init: bool = False
    norm_type: str = "rmsnorm"
    attn_proj: bool = False  # Whether to use attention projection

    # QK norm
    qk_norm: bool = False  # Apply norm after Q and K matmul
    norm_reorder: bool = False  # Apply rms-norm on output of attention instead of input
    trainable_rmsnorm: bool = True  # Allow rms norm weights to be trainable

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name != "expert_name":
            return

        # Define number of layers based on the stage
        num_hidden_layers = self.num_hidden_layers
        if self.expert_name == "lm_head" and self.num_hidden_layers_head is not None:
            num_hidden_layers = self.num_hidden_layers_head
        elif self.expert_name == "lm_body" and self.num_hidden_layers_body is not None:
            num_hidden_layers = self.num_hidden_layers_body
        elif self.expert_name == "lm_tail" and self.num_hidden_layers_tail is not None:
            num_hidden_layers = self.num_hidden_layers_tail

        super().__setattr__("num_hidden_layers", num_hidden_layers)

        if self.replace_n_layers:
            super().__setattr__("n_layers", num_hidden_layers)

    @computed_field
    @property
    def input_schema(self) -> dict[str, Any]:
        if self.expert_name is None:
            raise ValueError("expert_name must be set in LlamaArguments to determine input schema.")

        if self.expert_name == "lm_head":
            return {"sequence_length": self.max_seq_len}

        hid_dim = self.hidden_dim

        if self.use_compression:
            compression_length = int(self.hidden_dim // self.compression_rate)
            hid_dim = compression_length + 1  # +1 for tokens

        return {"sequence_length": self.max_seq_len, "hid_dim": hid_dim}


ModelArgumentsRegistry.register("llama", "default", LlamaArguments())

ModelArgumentsRegistry.register(
    "llama",
    "debugmodel",
    LlamaArguments(
        hidden_dim=256,
        n_layers=8,
        n_heads=16,
        rope_theta=500000,
        num_hidden_layers=2,
    ),
)

ModelArgumentsRegistry.register(
    "llama",
    "debugmodel_C",
    LlamaArguments(
        hidden_dim=256,
        n_layers=8,
        n_heads=16,
        rope_theta=500000,
        num_hidden_layers=2,
        qk_norm=True,
        norm_reorder=True,
        trainable_rmsnorm=False,
        compression_rate=5,
        use_compression=True,
        ss_component_path="s3://protocol-artifacts/subspace_compression/debugmodel_C/subspace_comp.pt",
    ),
)

ModelArgumentsRegistry.register(
    "llama",
    "debugmodel_C_pithos",
    LlamaArguments(
        hidden_dim=256,
        n_layers=8,
        n_heads=16,
        rope_theta=500000,
        num_hidden_layers=2,
        vocab_size=128256,
        qk_norm=True,
        norm_reorder=True,
        trainable_rmsnorm=False,
        compression_rate=5,
        use_compression=True,
        ss_component_path="s3://protocol-artifacts/subspace_compression/debugmodel_C/subspace_comp.pt",
    ),
)

ModelArgumentsRegistry.register(
    "llama",
    "400M",
    LlamaArguments(
        hidden_dim=1024,
        n_layers=20,
        n_heads=16,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=768,
        rope_theta=500000,
        num_hidden_layers=5,
    ),
)

ModelArgumentsRegistry.register(
    "llama",
    "400M_C",
    LlamaArguments(
        hidden_dim=1024,
        n_layers=20,
        n_heads=16,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=768,
        rope_theta=500000,
        num_hidden_layers=5,
        compression_rate=10,
        use_compression=True,
        ss_component_path="s3://protocol-artifacts/subspace_compression/400M_C/subspace_comp.pt",
    ),
)

ModelArgumentsRegistry.register(
    "llama",
    "500M",
    LlamaArguments(
        hidden_dim=1536,
        n_layers=16,
        n_heads=16,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        max_seq_len=1024,
        num_hidden_layers=4,
        qk_norm=True,
        norm_reorder=True,
        trainable_rmsnorm=False,
    ),
)

ModelArgumentsRegistry.register(
    "llama",
    "500M_C",
    LlamaArguments(
        hidden_dim=1536,
        n_layers=16,
        n_heads=16,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        max_seq_len=1024,
        num_hidden_layers=4,
        qk_norm=True,
        norm_reorder=True,
        trainable_rmsnorm=False,
        compression_rate=50,
        use_compression=True,
        ss_component_path="s3://protocol-artifacts/subspace_compression/500M_C/subspace_comp.pt",
    ),
)

ModelArgumentsRegistry.register(
    "llama",
    "1B_deep",
    LlamaArguments(
        hidden_dim=2048,
        n_layers=16,
        n_heads=16,
        n_kv_heads=8,
        max_seq_len=2048,
        ffn_dim_multiplier=1.3,
        multiple_of=768,
        rope_theta=500000,
        num_hidden_layers=1,
    ),
)

ModelArgumentsRegistry.register(
    "llama",
    "1B_C_deep",
    LlamaArguments(
        hidden_dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        max_seq_len=2048,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        num_hidden_layers=1,
        compression_rate=50,
        use_compression=True,
        ss_component_path="s3://protocol-artifacts/subspace_compression/1B_C_deep/subspace_comp.pt",
    ),
)

ModelArgumentsRegistry.register(
    "llama",
    "1B_C",
    LlamaArguments(
        hidden_dim=4096,
        n_layers=4,
        n_heads=32,
        n_kv_heads=8,
        max_seq_len=2048,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        num_hidden_layers=1,
        qk_norm=True,
        norm_reorder=True,
        trainable_rmsnorm=False,
        compression_rate=100,
        use_compression=True,
        ss_component_path="s3://protocol-artifacts/subspace_compression/8B_C/subspace_comp.pt",
    ),
)

ModelArgumentsRegistry.register(
    "llama",
    "4B_C",
    LlamaArguments(
        hidden_dim=4096,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        max_seq_len=2048,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        num_hidden_layers=1,
        compression_rate=100,
        use_compression=True,
        ss_component_path="s3://protocol-artifacts/subspace_compression/4B_C/subspace_comp.pt",
        attn_proj=True,
    ),
)

ModelArgumentsRegistry.register(
    "llama",
    "8B",
    LlamaArguments(
        hidden_dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        num_hidden_layers=1,
    ),
)

ModelArgumentsRegistry.register(
    "llama",
    "8B_C",
    LlamaArguments(
        hidden_dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        max_seq_len=4096,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        num_hidden_layers=1,
        qk_norm=True,
        norm_reorder=True,
        trainable_rmsnorm=False,
        compression_rate=100,
        use_compression=True,
        ss_component_path="s3://protocol-artifacts/subspace_compression/8B_C/subspace_comp.pt",
    ),
)

ModelArgumentsRegistry.register(
    "llama",
    "12B_C",
    LlamaArguments(
        hidden_dim=5120,
        n_layers=32,
        n_heads=40,
        n_kv_heads=8,
        max_seq_len=2048,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        num_hidden_layers_head=6,
        num_hidden_layers_body=5,
        num_hidden_layers_tail=6,
        vocab_size=128256,
        qk_norm=True,
        norm_reorder=True,
        trainable_rmsnorm=False,
        compression_rate=100,
        use_compression=True,
        ss_component_path="s3://protocol-artifacts/subspace_compression/12B_C_llama/subspace_comp.pt",
    ),
)

ModelArgumentsRegistry.register(
    "llama",
    "70B",
    LlamaArguments(
        hidden_dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000,
    ),
)

ModelArgumentsRegistry.register(
    "llama",
    "405B",
    LlamaArguments(
        hidden_dim=16384,
        n_layers=126,
        n_heads=128,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=4096,
        rope_theta=500000,
    ),
)
