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

from pydantic import computed_field

from agora_server.models.arguments_registry import ModelArgumentsRegistry
from agora_server.models.base_arguments import ModelArguments


class SimpleFCArguments(ModelArguments):
    max_seq_len: int = 32
    hidden_dim: int = 32
    vocab_size: int = 32
    # To artificially increase the stage size for AR testing, without increasing computation. Defaults to 1 (disabled).
    stage_size_multiplier: int = 1

    @computed_field
    @property
    def input_schema(self) -> dict[str, Any]:
        if self.expert_name is None:
            raise ValueError("expert_name must be set in SimpleFCArguments to determine input schema.")

        if self.expert_name == "lm_head":
            return {"sequence_length": self.max_seq_len}
        elif self.expert_name == "lm_body":
            return {"sequence_length": self.max_seq_len, "hid_dim": self.hidden_dim}
        else:
            return {"sequence_length": self.max_seq_len, "hid_dim": self.hidden_dim}


ModelArgumentsRegistry.register("simple_fc", "default", SimpleFCArguments())

ModelArgumentsRegistry.register(
    "simple_fc",
    "auto",
    SimpleFCArguments(
        max_seq_len=4096,
        hidden_dim=51,
        vocab_size=1,
        stage_size_multiplier=6,
    ),
)

ModelArgumentsRegistry.register(
    "simple_fc",
    "fc_1B",
    SimpleFCArguments(
        # Note: 1B params on CPU requires min 192GB RAM (eg m6a.12xlarge)
        max_seq_len=1,
        hidden_dim=4092,
        vocab_size=1,
        stage_size_multiplier=64,
    ),
)
