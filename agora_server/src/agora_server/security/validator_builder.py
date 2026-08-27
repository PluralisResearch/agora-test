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


from pydantic.v1 import BaseModel

from agora_server.hivemind.dht.crypto import RSASignatureValidator
from agora_server.hivemind.dht.schema import SchemaValidator
from agora_server.hivemind.dht.validation import RecordValidatorBase


class ValidatorBuilder:
    """Builder for creating lists of validators."""

    def __init__(self):
        self._validators: list[RecordValidatorBase] = []
        self._public_key = None

    def add_schema(self, schema: type[BaseModel], prefix: str | None = None) -> "ValidatorBuilder":
        """Add a validator with a custom schema."""
        self._validators.append(SchemaValidator(schema, prefix=prefix))
        return self

    def add_validator(self, validator: RecordValidatorBase) -> "ValidatorBuilder":
        """Add a custom validator instance."""
        self._validators.append(validator)
        return self

    def add_signature_validator(self) -> "ValidatorBuilder":
        """Add a RSA signature validator."""
        signature_validator = RSASignatureValidator()
        self._validators.append(signature_validator)
        self._public_key = signature_validator.local_public_key
        return self

    def build(self) -> tuple[list[RecordValidatorBase], bytes | None]:
        """Return the list of validators and public key."""
        return self._validators, self._public_key
