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

"""
Compression strategies that reduce the network communication in .averaging, .optim and .moe
"""

from agora_server.hivemind.compression.base import CompressionBase, CompressionInfo, NoCompression, TensorRole
from agora_server.hivemind.compression.floating import Float16Compression, ScaledFloat16Compression
from agora_server.hivemind.compression.quantization import (
    BlockwiseQuantization,
    Quantile8BitQuantization,
    Uniform8BitQuantization,
)
from agora_server.hivemind.compression.serialization import (
    deserialize_tensor_stream,
    deserialize_torch_tensor,
    serialize_torch_tensor,
)
