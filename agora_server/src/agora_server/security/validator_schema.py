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

from typing import TYPE_CHECKING

from pydantic.v1 import BaseModel

from agora_server.hivemind.dht.schema import BytesWithPublicKey


if TYPE_CHECKING:
    # For type checkers, treat it as bytes
    BytesWithPublicKeyType = bytes
else:
    # At runtime, use the actual validator
    BytesWithPublicKeyType = BytesWithPublicKey


class WorkerKillSignal(BaseModel):
    """Schema for worker kill signals."""

    peer_id: str


class WorkerKillSignalSchema(BaseModel):
    """Force kill signal keys to have signed subkeys."""

    terminate: dict[BytesWithPublicKeyType, WorkerKillSignal]
