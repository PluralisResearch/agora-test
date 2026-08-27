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


def get_flatten_greedy_dims(tensor: torch.Tensor, max_ndim: int = 2):
    """get dims to flatten tensor up to max_ndim dimensions by merging small axes together"""
    dims = list(tensor.shape)
    while len(dims) > max_ndim:
        squeeze_ix = min(range(len(dims) - 1), key=lambda i: dims[i] * dims[i + 1])
        squeezed_dim = dims.pop(squeeze_ix)
        dims[squeeze_ix] *= squeezed_dim
    return dims
