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

from abc import abstractmethod

import torch
import torch.nn.functional as F

from torch import nn

from agora_server.models.compression import _fit_fixed_token_weight
from agora_server.models.expert_registry import ExpertRegistry
from agora_server.models.reparam_llama.arguments import ReparamLlamaArguments
from agora_server.models.reparam_llama.components import (
    ReparametrizedEmbedding,
    TransformerBlock,
    build_norm,
    precompute_freqs_cis,
)


def head_sample_input(batch_size: int, sequence_length: int) -> torch.Tensor:
    return torch.randint(low=0, high=1000, size=(batch_size, sequence_length), dtype=torch.long)


def body_sample_input(batch_size: int, sequence_length: int, hid_dim: int) -> torch.Tensor:
    return torch.empty(batch_size, sequence_length, hid_dim)


def tail_sample_input(batch_size: int, sequence_length: int, hid_dim: int, fused_tail: bool = True):
    """Sample-input schema for the reparam_llama lm_tail expert.

    `fused_tail=True` (default) registers the 3-tensor schema
    (hidden, labels, loss_weight) consumed by TailModuleCollab. Set to False to fall
    back to the 2-tensor (hidden, labels) schema used with plain ModuleCollab.
    Server.create injects this kwarg via functools.partial based on its `fused_tail`
    parameter (sourced from the worker yaml config).
    """
    base = (
        torch.empty((batch_size, sequence_length, hid_dim)),
        torch.randint(0, 1000, (batch_size, sequence_length), dtype=torch.long),
    )
    if fused_tail:
        return (*base, torch.empty((batch_size, sequence_length)))
    return base


def dummy_tail_sample_input(batch_size: int, sequence_length: int, hid_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Dummy tail uses the original 2-tensor schema (no loss_weight) — it bypasses
    # TailModuleCollab.
    return (
        torch.empty((batch_size, sequence_length, hid_dim)),
        torch.randint(0, 1000, (batch_size, sequence_length), dtype=torch.long),
    )


class BaseExpert(nn.Module):
    """Base class for reparametrized Llama pipeline experts.

    Holds a single rcv buffer (shape: hidden_dim × rank) that serves as both:
      1. The shared orthonormal basis rcv for all reparametrized layers
         (W = Z @ rcv^T keeps each output weight in span(rcv)).
      2. The inter-stage compression / decompression matrix.

    One rcv per expert means VRAM for the basis scales with O(1) per stage,
    regardless of the number of layers.
    """

    def __init__(self, model_args: ReparamLlamaArguments, use_compression_per_layer: bool = True):
        super().__init__()
        self.model_args = model_args
        self.n_layers = model_args.n_layers

        if self.n_layers is None:
            raise ValueError("n_layers must be set in model_args before initializing the model.")

        if self.model_args.stage is None:
            raise ValueError("stage must be set in model_args before initializing the model.")

        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

        # Shared rcv = rcv used for both weight factorization and inter-stage compression
        self.compression_length = int(self.model_args.hidden_dim // self.model_args.compression_rate)
        self.fixed_tok_embeddings = nn.Embedding(self.model_args.vocab_size, self.model_args.hidden_dim)
        self.rcv = nn.Parameter(torch.empty(self.model_args.hidden_dim, self.compression_length), requires_grad=False)

        # forward_compression_list tracks which layers participate in the subspace
        # (used by ss_regularize interface; always no-op for reparametrized layers)
        self.forward_compression_list = [use_compression_per_layer] * self.n_layers

        # Compute cumulative layer offset for imbalanced/balanced stage allocations
        head_layers = self.model_args.num_hidden_layers_head or self.n_layers
        body_layers = self.model_args.num_hidden_layers_body or self.n_layers
        stage = self.model_args.stage
        layer_offset = 0 if stage == 0 else head_layers + (stage - 1) * body_layers

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(self.n_layers):
            layer_id_global = layer_offset + layer_id
            self.layers[str(layer_id)] = TransformerBlock(
                layer_id_global, self.model_args, use_compression=use_compression_per_layer
            )

    def initialize_layers(self, buffer_device: torch.device | None = None):
        """Initialize weights for all transformer blocks and fixed embeddings."""
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = self._precompute_freqs_cis()

        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights()

        nn.init.normal_(self.fixed_tok_embeddings.weight)
        self.fixed_tok_embeddings.weight.requires_grad = False

    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(
            self.model_args.hidden_dim // self.model_args.n_heads,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def compress_output(self, output: torch.Tensor) -> torch.Tensor:
        """Compress hidden states for inter-stage transmission.

        Expects output shape (B, S, hidden_dim + 1) where the last channel is
        the token id. Returns (B, S, rank + 1).
        """
        rcv = self.rcv.unsqueeze(0)

        x = output[:, :, :-1]
        idx = output[:, :, -1:]
        tokens = idx.to(torch.int).squeeze(2)

        fixed_embed = self.fixed_tok_embeddings(tokens)
        compressed = (rcv.transpose(2, 1) @ (x - fixed_embed).transpose(2, 1)).transpose(2, 1)
        return torch.cat([compressed, output[:, :, -1:]], dim=-1)

    def decompress_input(self, input: torch.Tensor) -> torch.Tensor:
        """Decompress hidden states received from the previous stage.

        Expects input shape (B, S, rank + 1). Returns (B, S, hidden_dim + 1).
        """
        rcv = self.rcv.unsqueeze(0)

        x = input[:, :, : self.compression_length].transpose(2, 1)
        idx = input[:, :, self.compression_length :]
        tokens = idx.to(torch.int).squeeze(2).clone()

        fixed_embed = self.fixed_tok_embeddings(tokens)
        decompressed = (rcv @ x + fixed_embed.transpose(2, 1)).transpose(2, 1)
        return torch.cat([decompressed, input[:, :, -1:]], dim=-1)

    def load_comp(self, ss_comps: dict):
        """Load subspace components (rcv and fixed token embeddings).

        ss_comps must contain:
            "rcv"              – Tensor of shape (hidden_dim, rank)
            "fixed_tok_weight" – Tensor of shape (vocab_size, hidden_dim)
        """
        self.rcv.data = ss_comps["rcv"].clone()
        fixed_tok_weight = _fit_fixed_token_weight(
            ss_comps["fixed_tok_weight"],
            self.fixed_tok_embeddings.weight.shape,
            self.model_args.pad_fixed_token_embeddings,
        )
        self.fixed_tok_embeddings.weight.data = fixed_tok_weight.clone()
        self.fixed_tok_embeddings.weight.requires_grad = False

    def ss_regularize(self):
        """No-op: reparametrized layers are in-subspace by construction."""
        return

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Each expert must implement its own forward method")


@ExpertRegistry.register("reparam_llama", "lm_head", head_sample_input)
class HeadExpert(BaseExpert):
    def __init__(self, model_args: ReparamLlamaArguments):
        super().__init__(model_args, use_compression_per_layer=True)

        rank = int(model_args.hidden_dim // model_args.compression_rate)
        self.tok_embeddings = ReparametrizedEmbedding(model_args.vocab_size, model_args.hidden_dim, rank)

        self.init_weights()

    def init_weights(self):
        self.initialize_layers()
        self.tok_embeddings.init_weights(init_std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # tok_embeddings projects ids into hidden space via rcv = self.rcv
        hidden_states = self.tok_embeddings(input_ids, self.rcv)
        hidden_states = hidden_states + self.fixed_tok_embeddings(input_ids)

        for layer in self.layers.values():
            hidden_states = layer(hidden_states, self.freqs_cis, rcv=self.rcv)

        # Append token ids and compress for the next stage
        hidden_states = torch.cat([hidden_states, input_ids.unsqueeze(2)], dim=-1)
        hidden_states = self.compress_output(hidden_states)
        return hidden_states


@ExpertRegistry.register("reparam_llama", "lm_body", body_sample_input)
class BodyExpert(BaseExpert):
    def __init__(self, model_args: ReparamLlamaArguments):
        super().__init__(model_args, use_compression_per_layer=True)
        self.initialize_layers()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.decompress_input(hidden_states)
        token_indices = hidden_states[:, :, -1:]
        hidden_states = hidden_states[:, :, :-1]

        for layer in self.layers.values():
            hidden_states = layer(hidden_states, self.freqs_cis, rcv=self.rcv)

        hidden_states = torch.cat([hidden_states, token_indices], dim=-1)
        hidden_states = self.compress_output(hidden_states)
        return hidden_states


@ExpertRegistry.register("reparam_llama", "lm_tail", tail_sample_input)
class TailExpert(BaseExpert):
    def __init__(self, model_args: ReparamLlamaArguments):
        # Tail layers are uncompressed (standard nn.Linear for wo and w2)
        super().__init__(model_args, use_compression_per_layer=False)
        self.forward_compression_list = [False] * model_args.n_layers

        self.norm = build_norm(model_args.norm_type, dim=model_args.hidden_dim, eps=model_args.norm_eps)
        self.output = nn.Linear(model_args.hidden_dim, model_args.vocab_size, bias=False)

        self.init_weights()

    def init_weights(self):
        self.initialize_layers()
        self.norm.reset_parameters()
        final_out_std = self.model_args.hidden_dim**-0.5
        cutoff_factor = 3
        nn.init.trunc_normal_(
            self.output.weight,
            mean=0.0,
            std=final_out_std,
            a=-cutoff_factor * final_out_std,
            b=cutoff_factor * final_out_std,
        )

    def forward(self, hidden_states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        hidden_states = self.decompress_input(hidden_states)
        hidden_states = hidden_states[:, :, :-1]  # strip token-id channel

        for layer in self.layers.values():
            hidden_states = layer(hidden_states, self.freqs_cis)

        hidden_states = self.norm(hidden_states)
        lm_logits = self.output(hidden_states)

        lm_logits = lm_logits.contiguous()
        labels = labels.contiguous()

        # reduction="none" is required: gradients must be sum-reduced (not mean-reduced) so that
        # GradientBuffer.accumulate_grads_ / populate_averaged_grads_ can correctly
        # normalize by total samples across varying batch sizes (from batch coalescing).
        loss = F.cross_entropy(lm_logits.permute(0, 2, 1), labels, reduction="none")
        return loss


@ExpertRegistry.register("reparam_llama", "lm_dummytail", dummy_tail_sample_input)
class DummyTailExpert(nn.Module):
    def __init__(self, model_args: ReparamLlamaArguments):
        super().__init__()
        self.model_args = model_args
        self.dummy = nn.Parameter(torch.ones(1, 1))

    def forward(self, hidden_states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return hidden_states[:, :, 0]

    def load_comp(self, ss_comps: dict):
        pass

    def ss_regularize(self):
        pass
