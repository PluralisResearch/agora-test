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

from collections.abc import Callable, Iterable
from enum import Enum
from typing import Any

import torch


Parameters = Iterable[torch.Tensor]
ParamGroups = Iterable[dict[str, Any]]
TorchOptimizer = torch.optim.Optimizer
LRSchedulerBase = torch.optim.lr_scheduler.LRScheduler

TorchOptimizerFactory = Callable[[Parameters | ParamGroups], TorchOptimizer]
SchedulerFactory = Callable[[TorchOptimizer], LRSchedulerBase]


class GhostPhase(Enum):
    """Ghost mode phases for joining workers with stale state.

    OFF: Normal operation — worker processes batches and contributes to AR.
    PHASE1: Receive-only — worker is invisible to trainers (no batches), participates in
        state AR with weight=0 to sync weights. Equivalent to the original ghost mode.
    PHASE2: Warm-up — worker becomes visible to trainers and processes batches (FW/BW),
        accumulating real gradients to warm up optimizer momentum/variance. Still does not
        contribute to AR (weight=0) or progress tracking (reports 0 samples).
    """

    OFF = 0
    PHASE1 = 1
    PHASE2 = 2


class ServerCreationError(Exception):
    """Raised when server creation is interrupted by a shutdown signal."""
