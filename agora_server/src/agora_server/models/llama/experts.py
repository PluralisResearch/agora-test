# This file contains code originally from Hivemind under MIT License
# Original: Copyright 2020 Learning@home authors and collaborators
# Modified by: Pluralis Research 2026
#
# Original code: MIT License (see THIRD_PARTY_LICENSES)
# Modifications: Apache 2.0 License (see LICENSE)
#
# Licensed under the Apache License, Version 2.0 (the "License") for modifications only;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

from abc import abstractmethod

import torch
import torch.nn.functional as F

from torch import nn

from agora_server.models.compression import _fit_fixed_token_weight
from agora_server.models.expert_registry import ExpertRegistry
from agora_server.models.llama.arguments import LlamaArguments
from agora_server.models.llama.components import TransformerBlock, build_norm, precompute_freqs_cis


def head_sample_input(batch_size: int, sequence_length: int) -> torch.Tensor:
    return torch.randint(low=0, high=1000, size=(batch_size, sequence_length), dtype=torch.long)


def body_sample_input(batch_size: int, sequence_length: int, hid_dim: int) -> torch.Tensor:
    return torch.empty(batch_size, sequence_length, hid_dim)


def tail_sample_input(batch_size: int, sequence_length: int, hid_dim: int, fused_tail: bool = True):
    """Sample-input schema for the lm_tail expert.

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
    # The dummy tail does not go through TailModuleCollab — it is used for tests/stage
    # isolation. Its schema stays at 2 tensors (no loss_weight) to keep the dummy
    # trainer path simple.
    return (
        torch.empty((batch_size, sequence_length, hid_dim)),
        torch.randint(0, 1000, (batch_size, sequence_length), dtype=torch.long),
    )


class BaseExpert(nn.Module):
    def __init__(self, model_args: LlamaArguments):
        super().__init__()
        self.model_args = model_args
        self.n_layers = model_args.n_layers

        if self.n_layers is None:
            raise ValueError("n_layers must be set in model_args before initializing the model.")

        if self.model_args.stage is None:
            raise ValueError("stage must be set in model_args before initializing the model.")

        # Precompute frequency tensor for positional embeddings
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

        # Initialize common compression components
        if self.model_args.use_compression:
            self.forward_compression_list = [True] * self.n_layers

            self.compression_length = int(self.model_args.hidden_dim // self.model_args.compression_rate)
            self.fixed_tok_embeddings = nn.Embedding(self.model_args.vocab_size, self.model_args.hidden_dim)
            self.rcv = nn.Parameter(
                torch.empty(self.model_args.hidden_dim, self.compression_length), requires_grad=False
            )

        # Compute cumulative layer offset for imbalanced/balanced stage allocations
        head_layers = self.model_args.num_hidden_layers_head or self.n_layers
        body_layers = self.model_args.num_hidden_layers_body or self.n_layers
        stage = self.model_args.stage
        layer_offset = 0 if stage == 0 else head_layers + (stage - 1) * body_layers

        # Layers dictionary
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(self.n_layers):
            layer_id_global = layer_offset + layer_id
            self.layers[str(layer_id)] = TransformerBlock(layer_id_global, self.model_args)

    def initialize_layers(self, buffer_device: torch.device | None = None):
        """Initialize weights for the base expert model.

        All models store precomputed freqs_cis and transformer blocks.
        """
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = self._precompute_freqs_cis()

        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights()

        # Initialize fixed token embeddings
        if self.model_args.use_compression and self.fixed_tok_embeddings is not None:
            nn.init.normal_(self.fixed_tok_embeddings.weight)
            self.fixed_tok_embeddings.weight.requires_grad = False

    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(
            self.model_args.hidden_dim // self.model_args.n_heads,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def compress_output(self, output):
        rcv = self.rcv.unsqueeze(0)

        # Extract output and token indices
        x = output[:, :, :-1]
        idx = output[:, :, -1:]
        tokens = idx.to(torch.int).squeeze(2)

        # Get fixed embeddings
        fixed_embed = self.fixed_tok_embeddings(tokens)

        # Compress: compressed ≈ rcv.T @ (output - fixed_embeddings)
        compressed_output = (rcv.transpose(2, 1) @ (x - fixed_embed).transpose(2, 1)).transpose(2, 1)

        # Concatenate token indices back to the compressed output
        return torch.cat([compressed_output, output[:, :, -1:]], dim=-1)

    def decompress_input(self, input):
        rcv = self.rcv.unsqueeze(0)

        # Extract compressed representation and token indices
        x = input[:, :, : self.compression_length].transpose(2, 1)
        idx = input[:, :, self.compression_length :]
        tokens = idx.to(torch.int).squeeze(2).clone()

        # Get fixed embeddings
        fixed_embed = self.fixed_tok_embeddings(tokens)

        # Decompress: h ≈ rcv @ compressed + fixed_embeddings
        decompressed_output = (rcv @ x + fixed_embed.transpose(2, 1)).transpose(2, 1)

        # Concatenate the tokens to the decompressed output
        return torch.cat([decompressed_output, input[:, :, -1:]], dim=-1)

    def load_comp(self, ss_comps):
        # Load RCV
        self.rcv.data = ss_comps["rcv"].clone()

        # Copy embedding weights
        fixed_tok_weight = _fit_fixed_token_weight(
            ss_comps["fixed_tok_weight"],
            self.fixed_tok_embeddings.weight.shape,
            self.model_args.pad_fixed_token_embeddings,
        )
        self.fixed_tok_embeddings.weight.data = fixed_tok_weight.clone()
        self.fixed_tok_embeddings.weight.requires_grad = False

    def ss_regularize(self):
        # Regularize attention and feedforward weights
        self.rcv.data = self.rcv.data.contiguous()
        with torch.no_grad():
            for i, layer in enumerate(self.layers.values()):
                if self.forward_compression_list[i]:
                    layer.attention.wo.weight.data = self.rcv @ (self.rcv.T @ layer.attention.wo.weight.data)
                    layer.attention.wo.weight.data = layer.attention.wo.weight.data.contiguous()
                    layer.feed_forward.w2.weight.data = self.rcv @ (self.rcv.T @ layer.feed_forward.w2.weight.data)
                    layer.feed_forward.w2.weight.data = layer.feed_forward.w2.weight.data.contiguous()

    @abstractmethod
    def forward(self, input_ids):
        """Abstract forward method that must be implemented by each child class.

        This ensures each expert handles its specific input processing.
        """
        raise NotImplementedError("Each expert must implement its own forward method")


@ExpertRegistry.register("llama", "lm_head", head_sample_input)
class HeadExpert(BaseExpert):
    def __init__(self, model_args: LlamaArguments):
        super().__init__(model_args)

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.hidden_dim)

        self.init_weights()

    def init_weights(self):
        # Initialize base layers
        self.initialize_layers()

        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)

    def ss_regularize(self):
        # For head expert call weight regularization and also regularize tok_embeddings
        super().ss_regularize()
        with torch.no_grad():
            self.tok_embeddings.weight.data = (self.rcv @ (self.rcv.T @ self.tok_embeddings.weight.data.T)).T
            self.tok_embeddings.weight.data = self.tok_embeddings.weight.data.contiguous()

    def forward(self, input_ids):
        hidden_states = self.tok_embeddings(input_ids)

        if self.model_args.use_compression:
            hidden_states += self.fixed_tok_embeddings(input_ids)

        for layer in self.layers.values():
            hidden_states = layer(hidden_states, self.freqs_cis)

        if self.model_args.use_compression:
            hidden_states = torch.cat([hidden_states, input_ids.unsqueeze(2)], dim=-1)
            hidden_states = self.compress_output(hidden_states)

        return hidden_states


@ExpertRegistry.register("llama", "lm_body", body_sample_input)
class BodyExpert(BaseExpert):
    def __init__(self, model_args: LlamaArguments):
        super().__init__(model_args)

        self.initialize_layers()

    def forward(self, hidden_states):
        if self.model_args.use_compression:
            hidden_states = self.decompress_input(hidden_states)
            # Extract token indices for later use
            token_indices = hidden_states[:, :, -1:]
            hidden_states = hidden_states[:, :, :-1]

        for layer in self.layers.values():
            hidden_states = layer(hidden_states, self.freqs_cis)

        if self.model_args.use_compression:
            # Reattach token indices and compress
            hidden_states = torch.cat([hidden_states, token_indices], dim=-1)
            hidden_states = self.compress_output(hidden_states)

        return hidden_states


@ExpertRegistry.register("llama", "lm_tail", tail_sample_input)
class TailExpert(BaseExpert):
    def __init__(self, model_args: LlamaArguments):
        model_args.attn_proj = False  # Attention projection is false on tail
        super().__init__(model_args)

        # The last layers are not compressed
        if model_args.use_compression:
            # Disable regularization on tail expert
            self.forward_compression_list = [False] * model_args.n_layers

        self.norm = build_norm(model_args.norm_type, dim=model_args.hidden_dim, eps=model_args.norm_eps)

        self.output = nn.Linear(model_args.hidden_dim, model_args.vocab_size, bias=False)

        self.init_weights()

    def init_weights(self):
        # Initialize base layers
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

    def forward(self, hidden_states, labels):
        if self.model_args.use_compression:
            hidden_states = self.decompress_input(hidden_states)
            hidden_states = hidden_states[:, :, :-1]  # Remove tokens

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


@ExpertRegistry.register("llama", "lm_dummytail", dummy_tail_sample_input)
class DummyTailExpert(nn.Module):
    def __init__(self, model_args: LlamaArguments):
        super().__init__()
        self.model_args = model_args
        self.dummy = nn.Parameter(torch.ones(1, 1))

    def forward(self, hidden_states, labels):
        return hidden_states[:, :, 0]

    def load_comp(self, ss_comps):
        pass

    def ss_regularize(self):
        pass
