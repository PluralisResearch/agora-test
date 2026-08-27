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

import torch
import torch.nn as nn
import torch.nn.functional as F

from agora_server.hivemind.utils import get_logger
from agora_server.models.expert_registry import ExpertRegistry
from agora_server.models.simple_fc.arguments import SimpleFCArguments


logger = get_logger(__name__)


def head_sample_input(batch_size: int, sequence_length: int) -> torch.Tensor:
    return torch.randint(low=0, high=1000, size=(batch_size, sequence_length), dtype=torch.long)


def body_sample_input(batch_size: int, sequence_length: int, hid_dim: int) -> torch.Tensor:
    return torch.empty((batch_size, sequence_length, hid_dim))


def tail_sample_input(batch_size: int, sequence_length: int, hid_dim: int, fused_tail: bool = True):
    """Sample-input schema for the simple_fc lm_tail expert.

    `fused_tail=True` (default) returns the 3-tensor (hidden, labels, loss_weight)
    schema consumed by TailModuleCollab; False falls back to the 2-tensor
    (hidden, labels) schema used with plain ModuleCollab. Server.create binds
    this kwarg via functools.partial.
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


@ExpertRegistry.register("simple_fc", "lm_head", head_sample_input)
class HeadExpert(nn.Module):
    def __init__(self, model_args: SimpleFCArguments):
        super().__init__()
        self.model_args = model_args
        self.fc = nn.Linear(model_args.max_seq_len, model_args.hidden_dim)

    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        # output: (batch, seq_len, hidden_dim)
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # Repeat hidden states for each position in sequence
        hidden_states = input_ids.unsqueeze(1).expand(batch_size, seq_len, -1)  # Emulate embedding layer
        return self.fc(hidden_states.float())


@ExpertRegistry.register("simple_fc", "lm_body", body_sample_input)
class BodyExpert(nn.Module):
    def __init__(self, model_args: SimpleFCArguments):
        super().__init__()
        self.model_args = model_args
        self.layers = nn.ModuleList(
            [nn.Linear(model_args.hidden_dim, model_args.hidden_dim) for _ in range(model_args.stage_size_multiplier)]
        )

    def forward(self, hidden_states):
        # hidden_states: (batch, seq_len, hidden_dim)
        # output: (batch, seq_len, hidden_dim)

        # Only inference the first layer - if more exists, they're only to inflate the total stage size for testing AR.
        return self.layers[0](hidden_states)


@ExpertRegistry.register("simple_fc", "lm_tail", tail_sample_input)
class TailExpert(nn.Module):
    def __init__(self, model_args: SimpleFCArguments):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.fc = nn.Linear(model_args.hidden_dim, model_args.vocab_size)

    def forward(self, hidden_states, labels):
        # hidden_states: (batch, seq_len, hidden_dim)
        # labels: (batch, seq_len)
        # output: (batch, seq_len)
        logits = self.fc(hidden_states)
        # Clamping as a hack due to actual tokenizer much larger than set vocab_size
        labels = torch.clamp(labels, 0, self.vocab_size - 1)
        logits = logits.permute(0, 2, 1).contiguous()
        labels = labels.contiguous()
        # reduction="none" is required: gradients must be sum-reduced (not mean-reduced) so that
        # GradientBuffer.accumulate_grads_ / populate_averaged_grads_ can correctly
        # normalize by total samples across varying batch sizes (from batch coalescing).
        loss = F.cross_entropy(logits, labels, reduction="none")
        return loss


@ExpertRegistry.register("simple_fc", "lm_dummytail", dummy_tail_sample_input)
class DummyTailExpert(nn.Module):
    def __init__(self, model_args: SimpleFCArguments):
        super().__init__()
        self.model_args = model_args
        self.dummy = nn.Parameter(torch.ones(1, 1))

    def forward(self, hidden_states, labels):
        return hidden_states[:, :, 0]
