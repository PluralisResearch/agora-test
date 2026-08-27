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


class ReparamLlamaArguments(TransformerArgumentsBase):
    """Arguments for the reparametrized Llama model.

    This model always uses compression (use_compression=True). The rcv
    matrix is loaded via ss_component_path and shared across all layers in an
    expert, keeping per-expert VRAM for the basis at O(hidden_dim * rank).
    """

    model_config = {"validate_assignment": True}

    hidden_dim: int = 4096
    n_heads: int = 32
    n_layers: int = 32  # Total layers; overwritten per-stage when replace_n_layers=True
    num_hidden_layers: int = 4  # Layers per stage
    num_hidden_layers_head: int | None = Field(frozen=True, default=None)
    num_hidden_layers_body: int | None = Field(frozen=True, default=None)
    num_hidden_layers_tail: int | None = Field(frozen=True, default=None)
    replace_n_layers: bool = True
    n_kv_heads: int | None = None
    vocab_size: int = 50265
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    max_seq_len: int = 512
    attn_proj: bool = False

    depth_init: bool = True
    constant_init: bool = False
    norm_type: str = "rmsnorm"
    qk_norm: bool = False
    norm_reorder: bool = False
    trainable_rmsnorm: bool = True

    # Compression is fundamental to this architecture; default True
    use_compression: bool = True

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name != "expert_name":
            return

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
            raise ValueError("expert_name must be set in ReparamLlamaArguments to determine input schema.")

        if self.expert_name == "lm_head":
            return {"sequence_length": self.max_seq_len}

        # Body and tail receive compressed activations (rank dims) + appended token id
        compression_length = int(self.hidden_dim // self.compression_rate)
        hid_dim = compression_length + 1  # +1 for the token-id channel

        return {"sequence_length": self.max_seq_len, "hid_dim": hid_dim}


ModelArgumentsRegistry.register("reparam_llama", "default", ReparamLlamaArguments())

ModelArgumentsRegistry.register(
    "reparam_llama",
    "debugmodel",
    ReparamLlamaArguments(
        hidden_dim=256,
        n_layers=8,
        n_heads=16,
        rope_theta=500000,
        num_hidden_layers=2,
        qk_norm=True,
        norm_reorder=True,
        trainable_rmsnorm=False,
        compression_rate=5,
        ss_component_path="s3://protocol-artifacts/subspace_compression/reparam_debugmodel/subspace_comp.pt",
    ),
)

ModelArgumentsRegistry.register(
    "reparam_llama",
    "auto",
    ReparamLlamaArguments(
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
        ss_component_path="s3://protocol-artifacts/subspace_compression/500M_C/subspace_comp.pt",
    ),
)

ModelArgumentsRegistry.register(
    "reparam_llama",
    "500M",
    ReparamLlamaArguments(
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
        ss_component_path="s3://protocol-artifacts/subspace_compression/500M_C/subspace_comp.pt",
    ),
)

ModelArgumentsRegistry.register(
    "reparam_llama",
    "500M_pithos",
    ReparamLlamaArguments(
        hidden_dim=1536,
        n_layers=16,
        n_heads=16,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        vocab_size=128256,
        multiple_of=1024,
        rope_theta=500000,
        max_seq_len=1024,
        num_hidden_layers=4,
        qk_norm=True,
        norm_reorder=True,
        trainable_rmsnorm=False,
        compression_rate=50,
        ss_component_path="s3://protocol-artifacts/subspace_compression/500M_C/subspace_comp.pt",
    ),
)

ModelArgumentsRegistry.register(
    "reparam_llama",
    "1B_C",
    ReparamLlamaArguments(
        hidden_dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        max_seq_len=2048,
        ffn_dim_multiplier=1.3,
        vocab_size=128256,
        multiple_of=1024,
        rope_theta=500000,
        num_hidden_layers_head=3,
        num_hidden_layers_body=4,
        num_hidden_layers_tail=5,
        replace_n_layers=True,
        compression_rate=50,
        use_compression=True,
        qk_norm=True,
        norm_reorder=True,
        trainable_rmsnorm=False,
        ss_component_path="s3://protocol-artifacts/subspace_compression/1B_C_deep_llama/subspace_comp.pt",
    ),
)

ModelArgumentsRegistry.register(
    "reparam_llama",
    "5B_C",  # 8B total, 5B trainable
    ReparamLlamaArguments(
        hidden_dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        max_seq_len=4096,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        num_hidden_layers=8,
        vocab_size=128256,
        qk_norm=True,
        norm_reorder=True,
        trainable_rmsnorm=False,
        compression_rate=100,
        use_compression=True,
        ss_component_path="s3://protocol-artifacts/subspace_compression/8B_C_llama/subspace_comp.pt",
    ),
)

ModelArgumentsRegistry.register(
    "reparam_llama",
    "8B_C",  # 12B total, 8.6B trainable
    ReparamLlamaArguments(
        hidden_dim=5120,
        n_layers=32,
        n_heads=40,
        n_kv_heads=8,
        max_seq_len=4096,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        num_hidden_layers_head=6,
        num_hidden_layers_body=4,
        num_hidden_layers_tail=6,
        replace_n_layers=True,
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
    "reparam_llama",
    "13B_C",  # 19B total, 12.2B trainable
    ReparamLlamaArguments(
        hidden_dim=5120,
        n_layers=50,
        n_heads=40,
        n_kv_heads=8,
        max_seq_len=4096,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        num_hidden_layers_head=4,
        num_hidden_layers_body=4,
        num_hidden_layers_tail=2,
        replace_n_layers=True,
        vocab_size=128256,
        qk_norm=True,
        norm_reorder=True,
        trainable_rmsnorm=False,
        compression_rate=100,
        use_compression=True,
        ss_component_path="s3://protocol-artifacts/subspace_compression/13B_C_reparam_llama/subspace_comp.pt",
    ),
)
