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

import asyncio
import math

from typing import Any

import torch

from agora_server.core.all_reduce.ar_runner import AveragingMode
from agora_server.core.all_reduce.matchmaking import MatchmakingException
from agora_server.core.averaging.state_averager import TrainingStateAverager
from agora_server.hivemind.averaging.group_info import GroupInfo
from agora_server.hivemind.averaging.load_balancing import load_balance_peers
from agora_server.hivemind.compression import CompressionInfo
from agora_server.hivemind.utils import get_logger
from agora_server.hivemind.utils.asyncio import enter_asynchronously
from agora_server.hivemind.utils.tensor_descr import TensorDescriptor


GatheredData = Any
logger = get_logger(__name__)


class IndexSelector:
    def __init__(self, p):
        self.state = {}
        self.p = p

    def get_indices(self, param, curr_partition):
        return torch.ones(param.shape).bool()


class PartitionedIndexSelector(IndexSelector):
    def __init__(self, p, param):
        super().__init__(p)
        self.state[param] = {}
        self._set_partition(param)

    def _set_partition(self, param):
        param_state = self.state[param]
        param_state["num_partitions"] = min(math.ceil(1 / self.p), param.numel())
        param_state["partitions"] = (
            torch.rand(param.numel(), device=param.device).argsort().view(param.shape) % param_state["num_partitions"]
        )
        if param_state["num_partitions"] <= 256:
            param_state["partitions"] = param_state["partitions"].to(torch.uint8)

    def get_indices(self, param, curr_partition):
        curr_partition = curr_partition % self.state[param]["num_partitions"]
        indices = (self.state[param]["partitions"] == curr_partition).bool()

        return indices


class SpartaTrainingStateAverager(TrainingStateAverager):
    """A class that extends TrainingStateAverager to perform sparse state averaging using SPARTA."""

    def __init__(
        self,
        start: bool,
        sparse_avg: float = 0.0,
        average_state_every: int = 1,
        beta_k_correction: bool = False,
        **kwargs,
    ):
        super().__init__(start=False, **kwargs)

        self._request_timeout = kwargs["request_timeout"]
        self.sparse_avg = sparse_avg
        self.average_state_every = average_state_every
        self.beta_k_correction = beta_k_correction
        self._snapshot_epoch: int | None = None
        self.partition_selector = []
        self._last_sparse_info: list[tuple[int, torch.Tensor]] | None = None
        self._old_sparse_tensors: list[torch.Tensor] | None = None

        if beta_k_correction and not kwargs.get("delta_rule_averaging", False):
            logger.warning("beta_k_correction=True has no effect without delta_rule_averaging=True")

        if sparse_avg:
            self.set_sparta_partitions()

        if start:
            self.run_in_background(await_ready=True)

    def set_sparta_partitions(self):
        with self.get_tensors() as local_tensors:
            torch.manual_seed(1337)
            for _i, p in enumerate(local_tensors):
                self.partition_selector.append(PartitionedIndexSelector(self.sparse_avg, p))

    def _compute_sparse_indices(self, local_tensors):
        """Compute sparse indices deterministically from global_epoch and partition_selector.

        This method is called from both the main process (_save_pre_averaging_state) and the child
        process (_aggregate_with_group) and produces identical results because all inputs
        (global_epoch, partition_selector) are consistent across processes.

        Returns:
            list of (tensor_idx, sparse_idx) tuples for each tensor that participates in averaging.
        """
        epoch = self.global_epoch
        snapped_epoch = int(round(epoch / self.average_state_every))

        n_params = len(self.main_parameters)
        n_opt_tensors = n_params * len(self.opt_keys_for_averaging)

        sparse_info = []

        for i, p in enumerate(local_tensors):
            should_include = False

            # Determine if tensor should be included
            if i < n_params:
                if self.main_parameters[i].requires_grad:
                    should_include = True
            else:
                should_include = True

            # Only process tensors that should be included
            if should_include:
                # Optimizer state tensors reuse the partition of their corresponding model parameter
                if n_params <= i < n_params + n_opt_tensors:
                    param_idx = (i - n_params) % n_params
                    _idx = self.partition_selector[param_idx].get_indices(local_tensors[param_idx], snapped_epoch)
                else:
                    _idx = self.partition_selector[i].get_indices(p, snapped_epoch)
                sparse_info.append((i, _idx))

        return sparse_info

    @torch.no_grad()
    def _save_pre_averaging_state(self):
        """Compute sparse indices and save old tensors in the main process for delta rule."""
        if not self.sparse_avg:
            return super()._save_pre_averaging_state()

        with self.get_tensors() as local_tensors:
            self._last_sparse_info = self._compute_sparse_indices(local_tensors)
            if self.delta_rule_averaging:
                self._old_sparse_tensors = [local_tensors[tidx][sidx].clone() for tidx, sidx in self._last_sparse_info]
                if self.beta_k_correction:
                    self._snapshot_epoch = self.local_epoch

    @torch.no_grad()
    def _apply_averaging_results_(self):
        """Apply only the sparse subset that was actually averaged, not the full tensor set."""
        if self._last_sparse_info is None:
            return super()._apply_averaging_results_()

        with self.get_tensors() as averaged_tensors:
            local_tensors = list(self._local_tensors())

            if not self.delta_rule_averaging or self._old_sparse_tensors is None:
                for tensor_idx, sparse_idx in self._last_sparse_info:
                    local_tensors[tensor_idx][sparse_idx] = averaged_tensors[tensor_idx][sparse_idx]
            else:
                # Compute beta^k scales for optimizer state deltas
                beta_scales = {}
                if self.beta_k_correction and self._snapshot_epoch is not None:
                    k = self.local_epoch - self._snapshot_epoch
                    if k > 0:
                        beta_scales = self._compute_beta_scales(k)

                for i, (tensor_idx, sparse_idx) in enumerate(self._last_sparse_info):
                    new_vals = averaged_tensors[tensor_idx][sparse_idx]
                    delta = new_vals - self._old_sparse_tensors[i]
                    if tensor_idx in beta_scales:
                        delta *= beta_scales[tensor_idx]
                    local_tensors[tensor_idx][sparse_idx] += delta.to(
                        device=local_tensors[tensor_idx].device,
                        dtype=local_tensors[tensor_idx].dtype,
                    )

            # Clamp negative exp_avg_sq values that delta rule can produce
            # TODO: this is a hack to prevent negative exp_avg_sq values, we should find a better way to do this
            if self.delta_rule_averaging and len(self.opt_keys_for_averaging) >= 2:
                n_params = len(self.main_parameters)
                exp_avg_sq_start = n_params * 2  # skip params and exp_avg
                exp_avg_sq_end = n_params * 3
                for tensor_idx, _ in self._last_sparse_info:
                    if exp_avg_sq_start <= tensor_idx < exp_avg_sq_end:
                        local_tensors[tensor_idx].clamp_(min=0.0)

        self._last_sparse_info = None
        self._old_sparse_tensors = None
        self._snapshot_epoch = None

    def _compute_beta_scales(self, k: int) -> dict[int, float]:
        """Compute per-tensor beta^k scale factors for optimizer state deltas.

        Returns a dict mapping tensor index to scale factor. Model param and extra
        tensor indices are omitted (no scaling). Non-Adam optimizers return {}.
        Returns empty dict for k<=0 (beta^0 = 1.0, no scaling needed).
        """
        if k <= 0:
            return {}
        betas = self.optimizer.param_groups[0].get("betas")
        if betas is None:
            return {}

        key_to_beta = {"exp_avg": betas[0], "exp_avg_sq": betas[1]}
        n_params = len(self.main_parameters)
        scales = {}

        for key_idx, opt_key in enumerate(self.opt_keys_for_averaging):
            beta = key_to_beta.get(opt_key)
            if beta is None:
                continue
            scale = beta**k
            start = n_params * (1 + key_idx)
            end = n_params * (2 + key_idx)
            for idx in range(start, end):
                scales[idx] = scale

        return scales

    async def _aggregate_with_group(self, group_info: GroupInfo, min_vector_size: int, **kwargs) -> GatheredData:
        """Run sparse aggregation in a given group and update tensors in place, return gathered metadata."""
        try:
            bandwidths, mode_ids, user_gathered_bytes = zip(*map(self.serializer.loads, group_info.gathered))
            user_gathered = dict(zip(group_info.peer_ids, map(self.serializer.loads, user_gathered_bytes)))
            modes = tuple(map(AveragingMode, mode_ids))

            # compute optimal part sizes from peer bandwidths;
            download_bandwidths = [
                thr if mode != AveragingMode.CLIENT else 0.0 for thr, mode in zip(bandwidths, modes)
            ]
            peer_fractions = await asyncio.get_event_loop().run_in_executor(
                None, load_balance_peers, self.total_size, download_bandwidths, min_vector_size
            )

            async with enter_asynchronously(self.get_tensors()) as local_tensors:
                if self.sparse_avg > 0:
                    sparse_info = self._compute_sparse_indices(local_tensors)
                    tensor_idxs = [tidx for tidx, _ in sparse_info]
                    sparse_idxs = [sidx for _, sidx in sparse_info]
                    sparse_tensor = [local_tensors[tidx][sidx].contiguous() for tidx, sidx in sparse_info]

                    # Build tensor info using proper indexing
                    tensor_infos_sparse = []
                    for sparse_idx, tensor_idx in enumerate(tensor_idxs):
                        desc = TensorDescriptor(
                            size=sparse_tensor[sparse_idx].shape,
                            dtype=sparse_tensor[sparse_idx].dtype,
                            device=sparse_tensor[sparse_idx].device,
                            requires_grad=local_tensors[tensor_idx].requires_grad,
                        )
                        tensor_infos_sparse.append(CompressionInfo(key=sparse_idx, descriptor=desc))

                    tensor_infos_sparse = tuple(tensor_infos_sparse)
                    kwargs["tensor_infos"] = tensor_infos_sparse

                    try:
                        await self._run_allreduce_inplace_(
                            sparse_tensor, group_info, peer_fractions=peer_fractions, modes=modes, **kwargs
                        )
                    finally:
                        # Copy results back using proper indexing
                        # Even if the await times out this copy back is applied
                        for sparse_idx, tensor_idx in enumerate(tensor_idxs):
                            local_tensors[tensor_idx][sparse_idxs[sparse_idx]] = sparse_tensor[sparse_idx]
                else:
                    await self._run_allreduce_inplace_(
                        local_tensors, group_info, peer_fractions=peer_fractions, modes=modes, **kwargs
                    )
                return user_gathered
        except BaseException as e:
            if isinstance(e, Exception):
                logger.error(e, exc_info=True)
            raise MatchmakingException(f"Unable to run All-Reduce: {type(e).__name__}: {e}") from e
