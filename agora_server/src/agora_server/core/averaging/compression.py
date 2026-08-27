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

"""BFloat16 compression for network transfer during gradient/state averaging.

This module provides BF16 compression that reduces network bandwidth by ~50%
while maintaining numerical stability for gradient averaging.

The BF16 compression handler is auto-registered with hivemind on module import,
so peers can deserialize BF16-compressed tensors automatically.
"""

import numpy as np
import torch

from agora_server.hivemind.compression.base import CompressionBase, CompressionInfo
from agora_server.hivemind.compression.floating import Float16Compression
from agora_server.hivemind.compression.serialization import _BASE_COMPRESSION_TYPES
from agora_server.hivemind.proto import runtime_pb2


class BFloat16Compression(CompressionBase):
    """Compresses tensors to bfloat16 for network transfer, preserving FP32 locally.

    BFloat16 has the same dynamic range as FP32 (8-bit exponent) but reduced
    precision (7-bit mantissa vs 23-bit). This makes it ideal for gradient
    compression where range matters more than precision.

    Unlike Float16Compression, no clamping is needed since BF16 can represent
    the same range as FP32.
    """

    # Reuse FLOAT16 compression type since we can't easily add new protobuf types
    # The dtype field will indicate the original dtype for proper reconstruction
    compression_type = runtime_pb2.CompressionType.FLOAT16

    def compress(self, tensor: torch.Tensor, info: CompressionInfo, allow_inplace: bool = False) -> runtime_pb2.Tensor:
        """Compress tensor to bfloat16 for network transfer.

        Args:
            tensor: Input tensor (typically FP32)
            info: Compression metadata
            allow_inplace: Whether in-place modification is allowed

        Returns:
            Serialized tensor in bfloat16 format
        """
        if not torch.is_floating_point(tensor):
            raise ValueError(f"{self.__class__.__name__} does not support {tensor.dtype} tensors")

        requires_grad = tensor.requires_grad
        tensor = tensor.detach().cpu()
        original_dtype_name = str(tensor.dtype).replace("torch.", "")

        # Convert to bfloat16 - no clamping needed since BF16 has FP32 range
        tensor_bf16 = tensor.to(torch.bfloat16)

        # BF16 doesn't have native numpy support, so we view as uint16 for serialization
        # This preserves the exact bit pattern
        tensor_as_uint16 = tensor_bf16.view(torch.uint16)

        return runtime_pb2.Tensor(
            compression=self.compression_type,
            buffer=tensor_as_uint16.numpy().tobytes(),
            size=tensor.shape,
            dtype=f"bf16_from_{original_dtype_name}",  # Custom marker for BF16 compression
            requires_grad=requires_grad,
        )

    def extract(self, serialized_tensor: runtime_pb2.Tensor) -> torch.Tensor:
        """Decompress bfloat16 tensor back to original dtype.

        Args:
            serialized_tensor: Serialized tensor from network

        Returns:
            Tensor restored to original dtype (typically FP32)
        """
        # Parse the original dtype from our marker
        dtype_str = serialized_tensor.dtype
        if dtype_str.startswith("bf16_from_"):
            original_dtype_str = dtype_str.replace("bf16_from_", "")
            original_dtype = getattr(torch, original_dtype_str)
        else:
            # Fallback for compatibility
            original_dtype = torch.float32

        # Deserialize as uint16 (BF16 bit pattern)
        array = np.frombuffer(serialized_tensor.buffer, dtype=np.uint16)
        tensor_uint16 = torch.as_tensor(array.copy()).reshape(tuple(serialized_tensor.size))

        # View as bfloat16 and convert to original dtype
        tensor_bf16 = tensor_uint16.view(torch.bfloat16)
        tensor = tensor_bf16.to(original_dtype)

        return tensor.requires_grad_(serialized_tensor.requires_grad)

    def estimate_compression_ratio(self, info: CompressionInfo) -> float:
        """Estimate compression ratio (compressed_bits / uncompressed_bits; lower = better compression)."""
        if info.descriptor.dtype == torch.float32:
            return 0.5  # 16 bits / 32 bits
        elif info.descriptor.dtype == torch.float64:
            return 0.25  # 16 bits / 64 bits
        else:
            return 1.0  # No compression benefit for other types


class Float16WithBFloat16Fallback(CompressionBase):
    """Wrapper that handles both standard FP16 and our custom BF16 compression.

    This class is registered with hivemind's deserialization system to handle
    tensors compressed with either format. It detects BF16 by the 'bf16_from_'
    prefix in the dtype field.
    """

    compression_type = runtime_pb2.CompressionType.FLOAT16

    def __init__(self):
        self._bf16 = BFloat16Compression()
        self._fp16 = Float16Compression()

    def compress(self, tensor: torch.Tensor, info: CompressionInfo, allow_inplace: bool = False) -> runtime_pb2.Tensor:
        """Compress using standard FP16 (use BFloat16Compression directly for BF16)."""
        return self._fp16.compress(tensor, info, allow_inplace)

    def extract(self, serialized_tensor: runtime_pb2.Tensor) -> torch.Tensor:
        """Extract tensor, detecting BF16 vs FP16 by dtype marker."""
        if serialized_tensor.dtype.startswith("bf16_from_"):
            return self._bf16.extract(serialized_tensor)
        return self._fp16.extract(serialized_tensor)

    def estimate_compression_ratio(self, info: CompressionInfo) -> float:
        """Estimate compression ratio (same for both FP16 and BF16)."""
        return self._fp16.estimate_compression_ratio(info)


_bf16_registered = False


def register_bf16_compression():
    """Register BF16 compression handler with hivemind's deserialization system.

    This must be called before any peer communication that uses BF16 compression,
    otherwise hivemind won't know how to deserialize BF16-compressed tensors.

    Safe to call multiple times - registration only happens once.
    """
    global _bf16_registered
    if _bf16_registered:
        return

    # Replace the FLOAT16 handler with one that can handle both formats
    _BASE_COMPRESSION_TYPES["FLOAT16"] = Float16WithBFloat16Fallback()
    _bf16_registered = True


# Auto-register on module import so hivemind can deserialize BF16 tensors from peers
register_bf16_compression()
