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

import multiprocessing as mp
import random
import threading

from collections.abc import Callable, Sequence
from functools import partial
from time import perf_counter
from typing import Any

import torch

from agora_server.core.averaging.gradient_buffer import GradientBufferFactory
from agora_server.core.averaging.state_averager import TrainingStateAveragerFactory
from agora_server.core.coordination.training_coordinator import TrainingCoordinator
from agora_server.core.metrics_reporter import TrainerMetricsReporter
from agora_server.core.server.activation_cache import ActivationCache
from agora_server.core.server.dht_handler import DHTHandlerThread, get_experts
from agora_server.core.server.module_collab import ModuleCollab, TailModuleCollab, _get_autocast_context
from agora_server.core.server.runtime import Runtime
from agora_server.core.server.w2w_coordinator import NextHopLearner
from agora_server.core.server.w2w_dataplane import DirectW2WDriver
from agora_server.core.server.w2w_head_injector import (
    PithosBatchStream,
    PithosShardStream,
    W2WHeadInjector,
    W2WOriginLedger,
)
from agora_server.core.server.w2w_local_router import (
    W2WLocalRouter,
    W2WMembership,
    W2WMembershipFeeder,
    W2WReceiptMailbox,
    join_order_data_rank,
    successor_stage_prefix,
)
from agora_server.hivemind.dht import DHT
from agora_server.hivemind.moe.expert_uid import UID_DELIMITER
from agora_server.hivemind.moe.server.connection_handler import ConnectionHandler
from agora_server.hivemind.moe.server.module_backend import ModuleBackend
from agora_server.hivemind.proto.runtime_pb2 import CompressionType
from agora_server.hivemind.utils.logging import get_logger
from agora_server.hivemind.utils.tensor_descr import DUMMY_BATCH_SIZE, BatchTensorDescriptor
from agora_server.logging.log_monitor import LogMonitor
from agora_server.models.base_arguments import ModelArguments
from agora_server.models.expert_registry import ExpertRegistry
from agora_server.models.lr_schedule import schedule_name_to_scheduler
from agora_server.monitor.peer_visibility import PeerVisibilityMonitor
from agora_server.security.auth import AuthorizerType
from agora_server.types import ServerCreationError, TorchOptimizer, TorchOptimizerFactory
from agora_server.utils.node_info import log_speedtest_results
from agora_server.utils.subspace import load_ss_components


logger = get_logger(__name__)


def _compile_warmup(
    module: torch.nn.Module,
    sample_input_func: Callable,
    input_schema: dict,
    device: torch.device | str,
    use_mixed_precision: bool,
    expert_uid: str,
    num_warmup_batches: int = 1,
    expert_name: str | None = None,
) -> None:
    """Run dummy forward+backward passes to trigger torch.compile before the worker starts serving.

    This ensures every worker is already compiled when it joins the network, avoiding the cascading
    starvation problem where the first-compiled worker hogs all traffic from the heap-based balancer.
    """
    device = torch.device(device) if isinstance(device, str) else device

    for i in range(num_warmup_batches):
        t_start = perf_counter()

        # Generate dummy inputs matching the real serving schema
        # Zero out float tensors because sample_input_func may use torch.empty (uninitialized memory),
        # which can contain NaN/Inf and cause CUDA asserts in loss functions like cross_entropy.
        sample_input = sample_input_func(DUMMY_BATCH_SIZE, **input_schema)
        if not isinstance(sample_input, (list, tuple)):
            sample_input = (sample_input,)
        inputs = []
        for t in sample_input:
            if isinstance(t, torch.Tensor):
                t = t.to(device)
                if t.is_floating_point():
                    t.zero_()
                inputs.append(t)
            else:
                inputs.append(t)

        # The lm_tail's serving schema may include extra inputs (e.g., loss_weight under
        # fused_tail) that are consumed by the TailModuleCollab wrapper, NOT by the
        # underlying TailExpert.forward(hidden, labels). The compiled module here is the
        # raw expert, so slice to its 2-arg arity. No-op for non-fused tail (already 2)
        # and for non-tail experts.
        if expert_name == "lm_tail" and len(inputs) > 2:
            inputs = inputs[:2]

        # Forward warmup (matches ModuleBackend.forward / ModuleCollab.forward -- no_grad + autocast)
        t_fwd_start = perf_counter()
        with torch.no_grad():
            with _get_autocast_context(use_mixed_precision, device):
                outputs = module(*inputs)
        t_fwd = perf_counter() - t_fwd_start

        # Backward warmup (matches ModuleCollab.backward -- enable_grad + autocast on forward only)
        t_bwd_start = perf_counter()
        with torch.enable_grad():
            inputs_grad = [t.detach().requires_grad_(True) if t.is_floating_point() else t.detach() for t in inputs]
            with _get_autocast_context(use_mixed_precision, device):
                outputs = module(*inputs_grad)

            if not isinstance(outputs, (list, tuple)):
                outputs = (outputs,)
            outputs_flat = tuple(o for o in outputs if isinstance(o, torch.Tensor) and o.requires_grad)
            grad_tensors = [torch.ones_like(o) for o in outputs_flat]
            torch.autograd.backward(outputs_flat, grad_tensors=grad_tensors, create_graph=False, retain_graph=False)
        t_bwd = perf_counter() - t_bwd_start

        # Clean up -- no optimizer step, so weights are unchanged.
        # Use set_to_none=False to preserve grad buffer identity for torch.compile:
        # compiled backward graphs hold references to the original grad tensors,
        # and setting grad = None would cause them to write to freed/reused memory.
        module.zero_grad(set_to_none=False)
        if device.type == "cuda":
            torch.cuda.empty_cache()

        t_total = perf_counter() - t_start
        logger.info(
            f"torch.compile warmup batch {i + 1}/{num_warmup_batches} for {expert_uid}: "
            f"forward={t_fwd:.2f}s, backward={t_bwd:.2f}s, total={t_total:.2f}s"
        )


def _validate_trainerless_data_config(
    trainerless_data_mode: str,
    pithos_corpus: str | None,
    pithos_corpus_uri: str | None,
    pithos_shard_index: int | None,
    pithos_shard_count: int | None,
) -> None:
    """Fail fast on an invalid trainerless head data-plane configuration."""
    if trainerless_data_mode not in ("stream", "shard"):
        raise ValueError(f"unknown trainerless_data_mode {trainerless_data_mode!r}; expected 'stream' or 'shard'")
    if trainerless_data_mode == "shard":
        if pithos_corpus is None and pithos_corpus_uri is None:
            raise ValueError(
                "trainerless_data_mode='shard' requires pithos_corpus and/or pithos_corpus_uri "
                "(the authorizer supplies pithos_corpus_uri on authed runs)"
            )
        if pithos_shard_index is None:
            raise ValueError(
                "trainerless_data_mode='shard' requires an explicit pithos_shard_index; shard mode "
                "has no join-order fallback, pass pithos_shard_index=<head ordinal> at launch"
            )
        if pithos_shard_count is None:
            raise ValueError("trainerless_data_mode='shard' requires pithos_shard_count")
    elif pithos_corpus is None:
        raise ValueError("trainerless head workers require pithos_corpus")


class Server(threading.Thread):
    """Server allows you to host "experts" - pytorch subnetworks that can be accessed remotely by peers.

    After creation, a server should be started: see Server.run or Server.run_in_background.

    A working server does two things:
        - processes incoming forward/backward requests via Runtime (created by the server)
        - publishes updates to expert status every :update_period: seconds
    """

    def __init__(
        self,
        dht: DHT,
        module_backends: dict[str, ModuleBackend],
        training_coordinator: TrainingCoordinator,
        checkpoint_saver: Any | None = None,
        num_connection_handlers: int = 1,
        update_period: float = 30,
        expiration: float | None = None,
        start: bool = False,
        auxiliary: bool = False,
        activation_cache: ActivationCache | None = None,
        w2w_coordinator: NextHopLearner | None = None,
        w2w_forward_driver: DirectW2WDriver | None = None,
        w2w_origin_injector: W2WHeadInjector | None = None,
        w2w_membership_feeder: W2WMembershipFeeder | None = None,
        w2w_manager=None,
        **kwargs,
    ):
        """Initialize Server.

        Args:
            dht (DHT): An instance of hivemind.DHT. Server will use DHT for all network interactions.
            module_backends (dict[str, ModuleBackend]): A dict {expert uid (str) : ModuleBackend} for all experts hosted by this server.
            training_coordinator (TrainingCoordinator): The training coordinator driving this peer's participation in the swarm.
            checkpoint_saver (Any | None, optional): Checkpoint saver instance. Defaults to None.
            num_connection_handlers (int, optional): Maximum number of simultaneous requests. Please note that the default value of 1 is too small for normal functioning, we recommend 4 handlers per expert backend. Defaults to 1.
            update_period (float, optional): How often will server attempt to publish its state (i.e. experts) to the DHT; if dht is None, this parameter is ignored. Defaults to 30.
            expiration (float | None, optional): When server declares its experts to the DHT, these entries will expire after this many seconds. Defaults to None.
            start (bool, optional): If True, the server will immediately start as a background thread and returns control after server is ready (see .ready below). Defaults to False.
            checkpoint_dir (Path | None, optional): Directory to save checkpoints. Defaults to None.
            **kwargs: Additional keyword arguments passed to Runtime.
        """
        super().__init__()
        self.dht, self.module_backends, self.update_period = dht, module_backends, update_period
        self.training_coordinator = training_coordinator
        self.checkpoint_saver = checkpoint_saver
        self.auxiliary = auxiliary
        self.activation_cache = activation_cache
        self.w2w_coordinator = w2w_coordinator
        self.w2w_forward_driver = w2w_forward_driver
        self.w2w_origin_injector = w2w_origin_injector
        self.w2w_membership_feeder = w2w_membership_feeder
        self._w2w_manager = w2w_manager
        # stats_report_interval is forwarded to Runtime via kwargs; reuse it for the cache/hop stats lines.
        self._activation_cache_report_interval = float(kwargs.get("stats_report_interval", 60))
        self._stop_serving = threading.Event()
        self._startup_error: Exception | None = None
        self._fatal_error: Exception | None = None
        self._shutdown_started = threading.Event()
        self._shutdown_lock = threading.Lock()

        self.conn_handlers = [
            ConnectionHandler(
                dht,
                self.module_backends,
                activation_cache=activation_cache,
                w2w_coordinator=w2w_coordinator,
                w2w_forward_driver=w2w_forward_driver,
            )
            for _ in range(num_connection_handlers)
        ]
        self.runtime = Runtime(self.module_backends, **kwargs)

        if self.module_backends:
            self.dht_handler_thread = DHTHandlerThread(
                module_backends=self.module_backends,
                dht=self.dht,
                update_period=self.update_period,
                expiration=expiration,
                daemon=True,
            )

        if start:
            self.run_in_background(await_ready=True)

    @classmethod
    def create(
        cls,
        model_name: str,
        model_conf: ModelArguments,
        expert_name: str,
        training_coordinator_factory: Callable[..., TrainingCoordinator],
        torch_optim_factory: TorchOptimizer | TorchOptimizerFactory,
        weight_decay: float,
        bandwidth: float | None,
        state_avg_factory: TrainingStateAveragerFactory,
        grad_buffer_factory: GradientBufferFactory | None,
        scheduler: str,
        num_warmup_steps: int | None,
        num_training_steps: int | None,
        min_batch_size: int,
        max_batch_size: int,
        stats_report_interval: int,
        coalesce_batches: bool = True,
        min_lr_ratio: float = 0.1,
        update_period: float = 30,
        num_experts: int | None = None,
        expert_uids: list[str] | None = None,
        expert_pattern: str | None = None,
        no_optim_params: list[str] | None = None,
        no_decay_params: list[str] | None = None,
        num_handlers: int | None = None,
        seed_peer: bool = False,
        device: torch.device | str | None = None,
        enable_tf32: bool = False,
        use_mixed_precision: bool = False,
        initial_peers: Sequence[str] = (),
        dump_addrs: str | None = None,
        compression: CompressionType = CompressionType.NONE,
        custom_module_path: str | None = None,
        expiration: float | None = None,
        storage_access: dict | None = None,
        checkpoint_manager_factory: Callable[..., Any] | None = None,  # noqa: E501 using Any type here for now until we create an abstract checkpoint saver class
        load_checkpoint: bool = False,
        prom_monitor_callback: Callable[..., Any] | None = None,
        authorizer: AuthorizerType | None = None,
        use_peer_visibility_monitor: bool = False,
        log_monitor: LogMonitor | None = None,
        delay_range_forward_backward: tuple[float, float] = (0.0, 0.0),
        use_torch_compile: bool = False,
        compile_warmup_batches: int = 1,
        metadata_store_url: str | None = None,
        metadata_store_require_signed_reads: bool = False,
        metadata_store_compress_requests: bool = False,
        fused_tail: bool = True,
        worker_activation_cache: bool = False,
        activation_cache_max_entries: int = 256,
        activation_cache_max_bytes: int = 8 * 1024**3,
        activation_cache_ttl_s: float = 1200.0,
        worker_to_worker: bool = False,
        w2w_ingress_max_entries: int | None = None,
        w2w_forward_timeout: float = 225.0,
        w2w_backward_timeout: float = 562.5,
        w2w_coord_timeout: float = 30.0,
        trainerless: bool = False,
        trainerless_max_inflight: int = 1,
        trainerless_adaptive_inflight: bool = False,
        trainerless_occupancy_low: float = 4.0,
        trainerless_occupancy_high: float = 10.0,
        trainerless_prefetch_batches: int = 4,
        trainerless_batch_size: int = 1,
        trainerless_data_start_timeout: float = 300.0,
        pithos_corpus: str | None = None,
        pithos_manifest_identity: str | None = None,
        pithos_cache_dir: str | None = None,
        pithos_cache_budget_bytes: int | None = None,
        pithos_prefetch_depth: int = 0,
        pithos_sequence_length: int | None = None,
        pithos_seed: int = 42,
        pithos_stream_count: int = 256,
        trainerless_data_mode: str = "shard",
        pithos_shard_index: int | None = None,
        pithos_shard_count: int | None = None,
        pithos_shards_per_head: int = 1,
        pithos_corpus_uri: str | None = None,
        trainerless_statistics_expiration: float = 30.0,
        num_stages: int | None = None,
        experiment_prefix: str = "pluralis",
        start: bool = False,
        auxiliary: bool = False,
        **kwargs,
    ) -> "Server":
        """Instantiate a server similar to hivemind moe server but support collaborative optimisation.

        Args:
            model_name (str): The model name, e.g. "llama", "mnist", etc.
            model_conf (ModelArguments): Model configuration.
            expert_name (str): Expert type e.g. "lm_head", "lm_body", etc.
            training_coordinator_factory (Callable[..., TrainingCoordinator]): Factory for creating the training coordinator.
            torch_optimizer_factory: TorchOptimizer | TorchOptimizerFactory: Use this torch optimizer for training.
            weight_decay (float): Weight decay value for torch optimizer.
            bandwidth (float | None): If specified, this value represents the network bandwidth available to averager.
            state_avg_factory (TrainingStateAveragerFactory): Factory for creating state averager.
            grad_buffer_factory (GradientBufferFactory | None): Factory for creating gradient buffer.
            scheduler (str): If not `none`, the name of the expert LR scheduler.
            num_warmup_steps (int | None): The number of warmup steps for LR schedule.
            num_training_steps (int | None): The total number of steps for LR schedule.
            min_lr_ratio (float, optional): The minimum learning rate as a ratio of the initial learning rate. Defaults to 0.1.
            min_batch_size (int): Total num examples in the same batch will be greater than this value.
            max_batch_size (int): Total num examples in the same batch will not exceed this value.
            stats_report_interval (int): Interval between two reports of batch processing performance statistics.
            update_period (float, optional): Period for updating DHT. Defaults to 30.
            num_experts (int | None, optional): Run this many identical experts. Defaults to None.
            expert_uids (list[str] | None, optional): Spawn experts with these exact uids, overrides num_experts and expert_pattern. Defaults to None.
            expert_pattern (str | None, optional): A string pattern or a list of expert uids, example: myprefix.[0:32].[0:256] means "sample random experts between myprefix.0.0 and myprefix.255.255". Defaults to None.
            no_optim_params (list[str] | None, optional): List of parameter name substrings that should not be passed to optimizer. Defaults to None.
            no_decay_params (list[str] | None, optional): List of parameter name substrings that should not be weight-decayed. Defaults to None.
            num_handlers (int | None, optional): Server will use this many parallel processes to handle incoming requests. Defaults to None.
            seed_peer (bool, optional): Whether this is a seed peer. Defaults to False.
            device (torch.device | str | None, optional): All experts will use this device in torch notation; default: cuda if available else cpu. Defaults to None.
            enable_tf32 (bool, optional): Whether to enable TF32 precision on supported devices (see docs/mixed_precision.md). Defaults to False.
            use_mixed_precision (bool, optional): Whether to enable BF16 mixed precision (see docs/mixed_precision.md). Defaults to False.
            initial_peers (Sequence[str], optional): Multiaddrs of one or more active DHT peers (if you want to join an existing DHT). Defaults to ().
            dump_addrs (str | None, optional): Path to dump addresses. Defaults to None.
            compression (CompressionType, optional): If specified, use this compression to pack all inputs, outputs and gradients by all experts hosted on this server. For a more fine-grained compression, start server in python and specify compression for each BatchTensorProto in ModuleBackend for the respective experts. Defaults to CompressionType.NONE.
            custom_module_path (str | None, optional): Path to custom model experts. Defaults to None.
            expiration (float | None, optional): Expiration time for DHT records. Defaults to None.
            storage_access (dict | None, optional): Object storage client settings for loading
                subspace components (``load_ss_components`` kwargs: access keys, region,
                endpoint). Defaults to None.
            checkpoint_manager_factory (Callable[..., Any] | None, optional): Factory for creating checkpoint saver or loader. Defaults to None.
            load_checkpoint (bool, optional): Whether to load from checkpoint. Defaults to False.
            prom_monitor_callback (Callable[..., Any] | None, optional): Prometheus monitoring callback. Defaults to None.
            authorizer (AuthorizerType | None, optional): Authorizer instance, forwarded to DHT for RPC auth. Defaults to None.
            use_peer_visibility_monitor (bool, optional): Whether to use peer visibility monitoring. Defaults to False.
            log_monitor (LogMonitor | None, optional): Log monitor instance. Defaults to None.
            delay_range_forward_backward (tuple[float, float], optional): Delay range for forward/backward passes. Defaults to (0.0, 0.0).
            use_torch_compile (bool, optional): Whether to enable torch.compile for forward/backward call-sites. Defaults to False.
            compile_warmup_batches (int, optional): Number of warmup batches to run before serving. Defaults to 1.
            metadata_store_url (str, optional): Metadata Store API base URL. When set, KV coordination is routed to the Metadata Store instead of the DHT.
            metadata_store_require_signed_reads (bool, optional): Drop unsigned Metadata Store records on read instead of accepting them with a warning. Flipped fleet-wide together with the server's signature_enforcement=enforce. Defaults to False.
            fused_tail (bool, optional): If True, the lm_tail expert is served via TailModuleCollab — a single fused-forward RPC that returns (loss_per_token, grad_hidden) and skips the tail backward RPC. Other expert types are unaffected. Must match the trainer's `collaboration_args.fused_tail`. Defaults to True.
            worker_to_worker (bool, optional): Coordination plane. When True, the worker reads, validates, logs, and counts the next hop a w2w trainer teaches it in-band, without acting on it (data still relays through the trainer). A no-op when the trainer is legacy (no hop present). Must match the trainer's `collaboration_args.worker_to_worker`. Defaults to False.
            trainerless_data_mode (str, optional): Trainerless head data plane: "shard" (the supported default) reads only the head's assigned corpus chunk file(s), with the index and corpus URL assigned by the authorizer on authed runs or passed explicitly on manual ones; "stream" (retained, unsupported) strides one congruence class across the whole corpus. Defaults to "shard".
            pithos_shard_index (int | None, optional): Index of the chunk-file shard(s) this head owns. Required on trainerless heads in shard mode; there is no join-order fallback. Defaults to None.
            pithos_shard_count (int | None, optional): Total number of shards; must equal the corpus chunk count. Required in shard mode. Defaults to None.
            pithos_shards_per_head (int, optional): Consecutive blocks each head owns in shard mode. Defaults to 1.
            pithos_corpus_uri (str | None, optional): Direct public corpus URL (normally handed out by the authorizer; a launch extra on manual runs). Supplies the corpus location — the packaged registry entry named by pithos_corpus pins identity and reader overrides but may carry no URI. Defaults to None.
            start (bool, optional): If True, starts server right away and returns when server is ready for requests. Defaults to False.
            auxiliary (bool, optional): If True, run as a dedicated "reducer": register the expert in the DHT (to count toward averaging quorum) and run the averager in AUX mode via a permanently-ghost-1 optimizer, but do NOT load weights from peers, run the Runtime, or process forward/backward. Defaults to False.
            **kwargs: Any other params will be forwarded to DHT upon creation.

        Returns:
            Server: The created server instance.

        Raises:
            ServerCreationError: If creation is interrupted by KeyboardInterrupt (SIGINT).
        """

        # Verify uid parameters
        if not (
            (expert_pattern is None and num_experts is None and expert_uids is not None)
            or (num_experts is not None and expert_uids is None)
        ):
            logger.error(
                "Please provide either expert_uids *or* num_experts (possibly with expert_pattern), but not both"
            )
            raise ValueError("Invalid expert uid parameters")

        # Get expert class and sample input function from registry
        if custom_module_path:
            ExpertRegistry.add_custom_models(custom_module_path)

        try:
            expert_class, sample_input_func = ExpertRegistry.get_expert_info(model=model_name, expert_name=expert_name)
        except Exception as e:
            logger.error(f"Failed to create expert {expert_name} from registry: {e}")
            raise

        if expert_name == "lm_tail":
            sample_input_func = partial(sample_input_func, fused_tail=fused_tail)

        dht = None
        experts: dict[str, ModuleCollab] = {}
        training_coordinator = None

        try:
            # Connect to DHT
            # When a Metadata Store is configured we run WITHOUT a Kademlia DHT: a
            # standalone libp2p transport (SwarmP2P, dht_mode="none") wrapped by RedisDHT
            # for KV. Peer discovery is the Metadata Store address book + the P2P resolver
            # hook (DHT->Redis migration M4). initial_peers is intentionally not forwarded
            # — there is no DHT to bootstrap. Without a Metadata Store, the legacy DHT path
            # is unchanged.
            if metadata_store_url is not None:
                from agora_server.core.metadata_store_token import MetadataStoreTokenProvider
                from agora_server.core.redis_dht import RedisDHT
                from agora_server.core.swarm_p2p import SwarmP2P
                from agora_server.security.auth import load_identity_key

                # Managed mode: requests to the store carry the authorizer's
                # short-lived AccessToken (refreshed by the provider).
                token_provider = MetadataStoreTokenProvider(authorizer) if authorizer is not None else None
                p2p = SwarmP2P(
                    start=True,
                    startup_timeout=30,
                    metadata_store_url=metadata_store_url,
                    metadata_store_token_provider=token_provider,
                    metadata_store_require_signed_reads=metadata_store_require_signed_reads,
                    **kwargs,
                )
                # Sign Metadata Store envelopes with the same persisted identity the
                # p2pd daemon runs on (P2P.create generates the file if missing, and
                # SwarmP2P has already started above), so peer_id == multihash(signing
                # pubkey) holds by construction.
                identity_path = kwargs.get("identity_path")
                private_key = load_identity_key(identity_path) if identity_path else None
                if private_key is None:
                    logger.warning("No identity_path configured; Metadata Store writes will be unsigned")
                dht = RedisDHT(
                    metadata_store_url=metadata_store_url,
                    p2p=p2p,
                    private_key=private_key,
                    token_provider=token_provider,
                    require_signed_reads=metadata_store_require_signed_reads,
                    compress_requests=metadata_store_compress_requests,
                )
                logger.info("Using SwarmP2P (no DHT) + Metadata Store for coordination")
            else:
                dht = DHT(
                    initial_peers=initial_peers,
                    start=True,
                    startup_timeout=30,
                    authorizer=authorizer,
                    **kwargs,
                )
            visible_maddrs_str = [str(a) for a in dht.get_visible_maddrs()]
            logger.info(f"Running P2P node on {visible_maddrs_str}")

            # Connect to monitor
            if log_monitor:
                log_monitor.connect_dht(dht)

            if dump_addrs is not None:
                with open(dump_addrs, "w") as text_file:
                    text_file.write(visible_maddrs_str[-1])

            # Generate uids
            if expert_uids is None:
                expert_uids = []
                uids_to_generate = num_experts - len(expert_uids)
                if uids_to_generate > 0:
                    logger.info(f"Generating {uids_to_generate} expert uids from pattern {expert_pattern}")
                    expert_uids.extend(_generate_uids(uids_to_generate, expert_pattern, dht))

            # Add optional monitors
            if use_peer_visibility_monitor:
                stage = expert_uids[0].split(".")[0]
                stage_uids = stage + "."
                _ = PeerVisibilityMonitor(dht, [stage_uids])

            num_experts = len(expert_uids)
            num_handlers = num_handlers if num_handlers is not None else num_experts * 8
            device = device or ("cuda" if torch.cuda.is_available() else "cpu")

            # TF32 is significantly faster than FP32 on modern GPUs
            logger.info("Current matmul precision: " + torch.get_float32_matmul_precision())
            if enable_tf32 and device != "cpu":
                logger.info("Enabling TF32 matmul precision")
                torch.set_float32_matmul_precision("high")

            # BF16 mixed precision for higher throughput on modern GPUs (H100, L40S, etc.)
            if use_mixed_precision:
                if device == "cpu":
                    logger.warning("Mixed precision requested but device is CPU, disabling")
                    use_mixed_precision = False
                elif not torch.cuda.is_bf16_supported():
                    logger.warning("Mixed precision requested but BF16 not supported on this GPU, disabling")
                    use_mixed_precision = False
                else:
                    logger.info("BF16 mixed precision enabled for forward/backward passes")

            # Scheduler
            scheduler_cls = schedule_name_to_scheduler[scheduler]
            if scheduler_cls is not None:
                scheduler_cls = partial(
                    scheduler_cls,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps,
                    min_lr_ratio=min_lr_ratio,
                )

            # Initialize experts
            input_schema = model_conf.input_schema
            sample_input = sample_input_func(DUMMY_BATCH_SIZE, **input_schema)
            if isinstance(sample_input, Sequence):
                args_schema = tuple(BatchTensorDescriptor.from_tensor(arg, compression) for arg in sample_input)
            else:
                args_schema = (BatchTensorDescriptor.from_tensor(sample_input, compression),)

            logger.info("Initializing expert")
            for expert_uid in expert_uids:
                expert = expert_class(model_conf)

                # Monitor callbacks
                if prom_monitor_callback:
                    prom_monitor_callback(
                        expert,
                        model_conf,
                        use_mixed_precision=use_mixed_precision,
                        enable_tf32=enable_tf32,
                        fused_tail=fused_tail and expert_name == "lm_tail",
                    )

                # Select parameters for optimizer
                no_decay_params = no_decay_params or []
                no_optim_params = no_optim_params or []

                params = [
                    {
                        "params": [
                            p
                            for n, p in expert.named_parameters()
                            if not any(nd in n for nd in no_decay_params)
                            and not any(no in n for no in no_optim_params)
                        ],
                        "weight_decay": weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in expert.named_parameters()
                            if any(nd in n for nd in no_decay_params) and not any(no in n for no in no_optim_params)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

                params = [group for group in params if group["params"]]

                # Load subspace components if needed
                if model_conf.use_compression and model_conf.ss_component_path:
                    if not storage_access:
                        storage_access = {}

                    ss_comps = load_ss_components(
                        model_conf.ss_component_path,
                        **storage_access,
                    )
                    expert.load_comp(ss_comps)
                    logger.info("Succeeded loading remote subspace components")
                    expert.ss_regularize()

                optimizer_lock = mp.Lock()

                # The lm_tail expert uses a fused-forward backend that returns
                # (loss_per_token, grad_hidden) in a single RPC, eliminating the
                # tail's activation recomputation. We must declare outputs_schema
                # explicitly because the underlying module (TailExpert) returns only
                # loss_per_token; ModuleBackend's schema inference would otherwise
                # build a single-tensor schema and silently drop grad_hidden at
                # serialization. Other experts keep the standard ModuleCollab path.
                if fused_tail and expert_name == "lm_tail":
                    backend_cls = TailModuleCollab
                    # args_schema = (hidden_descr, labels_descr, loss_weight_descr).
                    # Build the loss descriptor by running TailExpert.forward(hidden, labels)
                    # once on dummy zeros — TailExpert.forward does NOT take loss_weight, so
                    # we drop it for inference. grad_hidden has the same shape/dtype as hidden.
                    with torch.no_grad():
                        dummy_hidden = args_schema[0].make_zeros(DUMMY_BATCH_SIZE)
                        dummy_labels = args_schema[1].make_zeros(DUMMY_BATCH_SIZE)
                        dummy_loss = expert(dummy_hidden, dummy_labels)
                    loss_descriptor = BatchTensorDescriptor.from_tensor(dummy_loss, compression)
                    grad_hidden_descriptor = args_schema[0]
                    outputs_schema = (loss_descriptor, grad_hidden_descriptor)
                else:
                    backend_cls = ModuleCollab
                    outputs_schema = None

                expert_use_torch_compile = use_torch_compile
                if expert_use_torch_compile and auxiliary:
                    # Reducers never run forward/backward, so we can disable compile / warmup
                    # ensures nothing is resurrected in optimizer
                    logger.info(f"Disabling torch.compile for auxiliary server {expert_uid}")
                    expert_use_torch_compile = False

                backend = backend_cls(
                    optimizer_lock=optimizer_lock,
                    name=expert_uid,
                    module=expert,
                    args_schema=args_schema,
                    outputs_schema=outputs_schema,
                    min_batch_size=min_batch_size,
                    max_batch_size=max_batch_size,
                    delay_range_forward_backward=delay_range_forward_backward,
                    use_mixed_precision=use_mixed_precision,
                    use_torch_compile=expert_use_torch_compile,
                )

                backend.module.to(device)

                if expert_use_torch_compile and compile_warmup_batches > 0:
                    try:
                        _compile_warmup(
                            module=backend._compiled_module,
                            sample_input_func=sample_input_func,
                            input_schema=input_schema,
                            device=device,
                            use_mixed_precision=use_mixed_precision,
                            expert_uid=expert_uid,
                            num_warmup_batches=compile_warmup_batches,
                            expert_name=expert_name,
                        )
                    except Exception as e:
                        logger.warning(
                            f"torch.compile warmup failed for {expert_uid}, falling back to lazy compilation: {e}"
                        )

                # TODO: Add Swarm LambWithGradientClipping from torch_optim
                training_coordinator = training_coordinator_factory(
                    model=expert,
                    optimizer_lock=optimizer_lock,
                    optimizer=torch_optim_factory,
                    params=params,
                    dht=dht,
                    run_id=expert_uid.split(UID_DELIMITER)[0],
                    scheduler=scheduler_cls,
                    state_avg_factory=state_avg_factory,
                    grad_buffer_factory=grad_buffer_factory,
                    bandwidth=bandwidth,
                    auxiliary=auxiliary,
                )

                # Assign the training coordinator to the backend immediately so cleanup can find it
                backend.optimizer = training_coordinator
                experts[expert_uid] = backend

                if seed_peer:
                    logger.info("Seed node - not loading weights from peers")
                elif auxiliary:
                    logger.info("Reducer node - not loading weights from peers (contributes weight 0)")
                elif load_checkpoint:
                    logger.info("Will load state from checkpoint, skipping loading from peers")
                else:
                    logger.info("Loading weights from peers")
                    training_coordinator.load_state_from_peers(wait_for_end_round=True)

                training_coordinator.tracker.allow_progress_report = True
                training_coordinator.set_averagers_allow_state_sharing()

            # Re-emit after the prom queue handler is gating in logs (start_logs flipped
            # by prom_monitor_callback above), so the static speedtest values land in Prometheus.
            if prom_monitor_callback is not None and authorizer is not None:
                log_speedtest_results(
                    authorizer.peer_info["download_speed"],
                    authorizer.peer_info["upload_speed"],
                    authorizer.peer_info["latency"],
                )

            if worker_to_worker and not fused_tail:
                raise ValueError("worker_to_worker requires fused_tail=True")
            if worker_to_worker and not worker_activation_cache and expert_name != "lm_tail":
                raise ValueError("worker_to_worker requires worker_activation_cache=True on head/body workers")
            if trainerless:
                if not worker_to_worker:
                    raise ValueError("trainerless requires worker_to_worker=True")
                if metadata_store_url is None:
                    raise ValueError("trainerless requires the Metadata Store (metadata_store_url)")
                if num_stages is None:
                    raise ValueError("trainerless requires num_stages to derive the successor stage")

            own_uid = expert_uids[0] if expert_uids else ""
            trainerless_batch_source = None
            trainerless_data_idx = None
            trainerless_head = trainerless and own_uid.startswith("head")
            if trainerless_head:
                required_pithos_values = (
                    ("pithos_manifest_identity", pithos_manifest_identity),
                    ("pithos_cache_dir", pithos_cache_dir),
                    ("pithos_cache_budget_bytes", pithos_cache_budget_bytes),
                    ("pithos_sequence_length", pithos_sequence_length),
                )
                for required_name, required_value in required_pithos_values:
                    if required_value is None:
                        raise ValueError(f"trainerless head workers require {required_name}")
                _validate_trainerless_data_config(
                    trainerless_data_mode,
                    pithos_corpus,
                    pithos_corpus_uri,
                    pithos_shard_index,
                    pithos_shard_count,
                )
                assert pithos_manifest_identity is not None
                assert pithos_cache_dir is not None
                assert pithos_cache_budget_bytes is not None
                assert pithos_sequence_length is not None
                if pithos_sequence_length != model_conf.max_seq_len:
                    raise ValueError(
                        f"pithos_sequence_length {pithos_sequence_length} does not match "
                        f"model max_seq_len {model_conf.max_seq_len}"
                    )

            # Optionally create checkpoint saver and load experts
            try:
                checkpoint_saver = checkpoint_manager_factory(experts) if checkpoint_manager_factory else None
            except Exception as e:
                logger.error(f"Failed to create checkpoint saver: {e}")
                raise

            if load_checkpoint and checkpoint_saver:
                try:
                    checkpoint_saver.load_experts()
                except Exception as e:
                    logger.error(f"Failed to load experts from checkpoint: {e}")
                    raise

            training_coordinator.wait_for_join_window()

            if trainerless_head:
                assert pithos_manifest_identity is not None
                assert pithos_cache_dir is not None
                assert pithos_cache_budget_bytes is not None
                assert pithos_sequence_length is not None
                if trainerless_data_mode == "shard":
                    assert pithos_shard_index is not None
                    assert pithos_shard_count is not None
                    trainerless_data_idx = pithos_shard_index
                    trainerless_batch_source = PithosShardStream(
                        corpus_name=pithos_corpus,
                        corpus_uri=pithos_corpus_uri,
                        expected_manifest_identity=pithos_manifest_identity,
                        cache_dir=pithos_cache_dir,
                        cache_budget_bytes=pithos_cache_budget_bytes,
                        prefetch_depth=pithos_prefetch_depth,
                        sequence_length=pithos_sequence_length,
                        seed=pithos_seed,
                        shard_index=pithos_shard_index,
                        shard_count=pithos_shard_count,
                        shards_per_head=pithos_shards_per_head,
                        batch_size=trainerless_batch_size,
                    )
                    logger.info(
                        f"Trainerless head {own_uid}: configured Pithos shard "
                        f"{pithos_shard_index}/{pithos_shard_count} (x{pithos_shards_per_head}), "
                        f"manifest {pithos_manifest_identity}; pithos_stream_count is ignored in shard mode"
                    )
                else:
                    assert pithos_corpus is not None
                    trainerless_data_idx = join_order_data_rank(dht, own_uid)
                    if trainerless_data_idx >= pithos_stream_count:
                        raise ValueError(
                            f"trainerless head rank {trainerless_data_idx} is outside configured "
                            f"pithos_stream_count {pithos_stream_count}"
                        )
                    trainerless_batch_source = PithosBatchStream(
                        corpus_name=pithos_corpus,
                        corpus_uri=pithos_corpus_uri,
                        expected_manifest_identity=pithos_manifest_identity,
                        cache_dir=pithos_cache_dir,
                        cache_budget_bytes=pithos_cache_budget_bytes,
                        prefetch_depth=pithos_prefetch_depth,
                        sequence_length=pithos_sequence_length,
                        seed=pithos_seed,
                        stream_count=pithos_stream_count,
                        data_idx=trainerless_data_idx,
                        batch_size=trainerless_batch_size,
                    )
                    logger.info(
                        f"Trainerless head {own_uid}: configured Pithos stream "
                        f"{trainerless_data_idx}/{pithos_stream_count}, manifest {pithos_manifest_identity}"
                    )

            activation_cache = None
            # The fused tail returns its gradient inside the forward RPC and never receives a backward
            # RPC, so there is nothing to serve back: a fused-tail worker simply gets no cache (and thus
            # no caching, reporter, or overhead). Head/body workers and a non-fused tail still get one.
            if worker_activation_cache and not (fused_tail and expert_name == "lm_tail"):
                activation_cache = ActivationCache(
                    max_entries=activation_cache_max_entries,
                    max_bytes=activation_cache_max_bytes,
                    ttl_seconds=activation_cache_ttl_s,
                    name=expert_uids[0] if expert_uids else "",
                )
                logger.info(
                    f"Worker activation cache enabled (max_entries={activation_cache_max_entries}, "
                    f"max_bytes={activation_cache_max_bytes}, ttl_s={activation_cache_ttl_s})"
                )
            elif worker_activation_cache:
                logger.info("Worker activation cache skipped for the fused-tail worker (no backward RPC to serve)")

            w2w_coordinator = None
            w2w_forward_driver = None
            w2w_origin_injector = None
            w2w_membership_feeder = None
            w2w_manager = None
            # Unlike the activation cache, the learner is instantiated on ALL w2w workers including the
            # fused tail: it is observational on rpc_forward (which the fused tail receives), and the
            # tail simply never carries a hop, so it no-ops naturally.
            if worker_to_worker:
                w2w_coordinator = NextHopLearner(name=expert_uids[0] if expert_uids else "")
                local_router = None
                origin_ledger = None
                receipt_mailbox = None
                own_peer_id = None
                if trainerless:
                    logger.info("Worker-to-worker enabled (trainerless: sender-local next-hop resolution)")
                    w2w_manager = mp.Manager()
                    receipt_mailbox = W2WReceiptMailbox(w2w_manager.dict())
                    successor = successor_stage_prefix(own_uid, num_stages)
                    if successor is not None:
                        membership = W2WMembership(w2w_manager.dict())
                        w2w_membership_feeder = W2WMembershipFeeder(
                            dht=dht, successor_prefix=successor, membership=membership, update_period=update_period
                        )
                        local_router = W2WLocalRouter(
                            successor_prefix=successor,
                            membership=membership,
                            mailbox=receipt_mailbox,
                            lease_ttl=w2w_forward_timeout + 60.0,
                        )
                    if own_uid.startswith("head"):
                        origin_ledger = W2WOriginLedger(w2w_manager.dict(), own_uid)
                        own_peer_id = dht.peer_id.to_base58()
                else:
                    logger.info("Worker-to-worker enabled (direct data path, trainer-mediated next-hop resolution)")
                w2w_forward_driver = DirectW2WDriver(
                    max_entries=w2w_ingress_max_entries or activation_cache_max_entries,
                    forward_timeout=w2w_forward_timeout,
                    backward_timeout=w2w_backward_timeout,
                    coord_timeout=w2w_coord_timeout,
                    name=expert_uids[0] if expert_uids else "",
                    trainerless=trainerless,
                    local_router=local_router,
                    origin_ledger=origin_ledger,
                    own_peer_id=own_peer_id,
                    receipt_mailbox=receipt_mailbox,
                )
                if origin_ledger is not None:
                    if trainerless_batch_source is None or trainerless_data_idx is None:
                        raise RuntimeError("trainerless head Pithos source was not initialized")
                    assert num_stages is not None
                    assert pithos_sequence_length is not None
                    reporter = TrainerMetricsReporter(
                        dht,
                        dht.local_public_key,
                        experiment_prefix,
                        trainerless_statistics_expiration,
                        trainerless_data_idx % num_stages,
                        trainerless_batch_size,
                        pithos_sequence_length,
                    )
                    w2w_origin_injector = W2WHeadInjector(
                        driver=w2w_forward_driver,
                        ledger=origin_ledger,
                        reporter=reporter,
                        dht=dht,
                        expert_uid=own_uid,
                        expert_backend=experts[own_uid],
                        batch_source=trainerless_batch_source,
                        num_stages=num_stages,
                        data_idx=trainerless_data_idx,
                        max_inflight=trainerless_max_inflight,
                        prefetch_batches=trainerless_prefetch_batches,
                        forward_timeout=w2w_forward_timeout,
                        backward_timeout=w2w_backward_timeout,
                        activation_cache=activation_cache,
                        adaptive_inflight=trainerless_adaptive_inflight,
                        occupancy_low=trainerless_occupancy_low,
                        occupancy_high=trainerless_occupancy_high,
                        data_start_timeout=trainerless_data_start_timeout,
                    )
                if w2w_membership_feeder is not None:
                    w2w_membership_feeder.start()

            return cls(
                dht=dht,
                module_backends=experts,
                training_coordinator=training_coordinator,
                checkpoint_saver=checkpoint_saver,
                w2w_origin_injector=w2w_origin_injector,
                w2w_membership_feeder=w2w_membership_feeder,
                w2w_manager=w2w_manager,
                num_connection_handlers=num_handlers,
                update_period=update_period,
                expiration=expiration,
                start=start,
                auxiliary=auxiliary,
                device=device,
                stats_report_interval=stats_report_interval,
                coalesce_batches=coalesce_batches,
                activation_cache=activation_cache,
                w2w_coordinator=w2w_coordinator,
                w2w_forward_driver=w2w_forward_driver,
            )

        except KeyboardInterrupt:
            logger.info("Server creation interrupted, cleaning up partially initialized components...")
            if training_coordinator is not None:
                logger.info("Shutting down training coordinator")
                training_coordinator.shutdown()
            if dht is not None:
                logger.info("Shutting down DHT")
                dht.shutdown()
            raise ServerCreationError("Server creation interrupted by shutdown signal") from None

    def run(self):
        """Start Server in the current thread.

        Initialize dht if necessary, start connection handlers, run Runtime (self.runtime) to process incoming requests.
        """
        logger.info(f"Server started with {len(self.module_backends)} modules:")
        for expert_name, backend in self.module_backends.items():
            num_parameters = sum(p.numel() for p in backend.module.parameters() if p.requires_grad)
            logger.info(f"{expert_name}: {backend.module.__class__.__name__}, {num_parameters} parameters")

        if not self.dht.is_alive():
            self.dht.run_in_background(await_ready=True)

        if self.module_backends:
            self.dht_handler_thread.start()

        if self.checkpoint_saver is not None and isinstance(self.checkpoint_saver, threading.Thread):
            self.checkpoint_saver.start()

        for handler in self.conn_handlers:
            handler.run_in_background()

        if self.activation_cache is not None:
            threading.Thread(
                target=self._run_activation_cache_reporter,
                name="activation-cache-reporter",
                daemon=True,
            ).start()

        if self.w2w_coordinator is not None:
            threading.Thread(
                target=self._run_w2w_coordinator_reporter,
                name="w2w-coordinator-reporter",
                daemon=True,
            ).start()

        if self.w2w_forward_driver is not None:
            threading.Thread(
                target=self._run_w2w_send_reporter,
                name="w2w-send-reporter",
                daemon=True,
            ).start()

        if self.w2w_origin_injector is not None:
            # Started here so the runtime, connection handlers, and optimizer join are all in
            # place before the first self-injected microbatch queues on the forward pool.
            try:
                self.w2w_origin_injector.start()
            except Exception as e:
                logger.error(f"Trainerless injector failed to start: {type(e).__name__}: {e}. Exiting run.")
                self._abort_startup(e)
                return
            threading.Thread(
                target=self._watch_w2w_origin_failure,
                name="w2w-origin-failure-watcher",
                daemon=True,
            ).start()

        self.training_coordinator.start_monitor()

        if self.auxiliary:
            self.runtime.ready.set()
            self._stop_serving.wait()
        else:
            self.runtime.run()

    def _abort_startup(self, error: Exception) -> None:
        """Record a startup failure, tear down, and release the ready waiter.

        Raising out of run() would only end the server thread and leave the process
        alive but never ready; run_in_background re-raises the recorded error instead.
        """
        self._startup_error = error
        try:
            self.shutdown()
        except Exception as e:
            logger.warning(f"Shutdown after startup failure raised: {type(e).__name__}: {e}")
        self.runtime.ready.set()

    def _watch_w2w_origin_failure(self) -> None:
        injector = self.w2w_origin_injector
        if injector is None:
            return
        error = injector.wait_for_terminal_failure()
        if error is None:
            return
        self._fatal_error = error
        logger.error(f"Trainerless data source failed terminally; stopping worker: {type(error).__name__}: {error}")
        self.shutdown()

    @property
    def fatal_error(self) -> Exception | None:
        """Terminal asynchronous failure that forced this server to stop."""
        return self._fatal_error

    def run_in_background(self, await_ready: bool = True, timeout: float | None = None):
        """Start Server in a background thread.

        If `await_ready`, this method will wait until background server is ready to process incoming requests or for `timeout` seconds max.
        """
        self.start()
        if await_ready:
            if not self.ready.wait(timeout=timeout):
                raise TimeoutError("Server didn't notify .ready in {timeout} seconds")
            if self._startup_error is not None:
                raise ServerCreationError("Server failed during startup") from self._startup_error

    @property
    def ready(self) -> mp.synchronize.Event:
        """An event (multiprocessing.Event) that is set when the server is ready to process requests.

        **Example:**

        >>> server.start()
        >>> server.ready.wait(timeout=10)
        >>> print("Server ready" if server.ready.is_set() else "Server didn't start in 10 seconds")
        """
        return self.runtime.ready  # mp.Event that is true if self is ready to process batches

    def _run_activation_cache_reporter(self):
        """Periodically log a cumulative ``[ActCache]`` stats line (scraped by WorkerPromMonitor and
        used to verify cache health / parity). Runs in a daemon thread on caching workers and stops
        when the server begins shutdown."""
        interval = self._activation_cache_report_interval
        while not self._stop_serving.wait(interval):
            try:
                s = self.activation_cache.stats()
                logger.info(
                    f"[ActCache] entries: {s['entries']}, bytes: {s['bytes']}, puts: {s['puts']}, "
                    f"hits: {s['hits']}, misses: {s['misses']}, evictions_fifo: {s['evictions_fifo']}, "
                    f"evictions_ttl: {s['evictions_ttl']}, overwrites: {s['overwrites']}, drops: {s['drops']}"
                )
            except Exception as e:
                logger.warning(f"[ActCache] failed to report stats: {type(e).__name__}")

    def _run_w2w_coordinator_reporter(self):
        """Periodically log a cumulative ``[W2WHop]`` stats line (scraped by WorkerPromMonitor and
        used to verify next-hop dissemination coherence). Runs in a daemon thread on w2w workers and
        stops when the server begins shutdown. Same in-process mechanism as the ActCache reporter."""
        interval = self._activation_cache_report_interval
        while not self._stop_serving.wait(interval):
            try:
                s = self.w2w_coordinator.stats()
                logger.info(
                    f"[W2WHop] next_hops_learned: {s['next_hops_learned']}, "
                    f"prev_hops_learned: {s['prev_hops_learned']}, parse_errors: {s['parse_errors']}"
                )
            except Exception as e:
                logger.warning(f"[W2WHop] failed to report stats: {type(e).__name__}")

    def _run_w2w_send_reporter(self):
        interval = self._activation_cache_report_interval
        while not self._stop_serving.wait(interval):
            try:
                s = self.w2w_forward_driver.stats()
                logger.info(
                    f"[W2WSend] entries: {s.get('entries', 0)}, accepted_forward: {s.get('accepted_forward', 0)}, "
                    f"accepted_backward: {s.get('accepted_backward', 0)}, busy: {s.get('busy', 0)}, "
                    f"forward_pushes: {s.get('forward_pushes', 0)}, backward_pushes: {s.get('backward_pushes', 0)}, "
                    f"busy_retries: {s.get('busy_retries', 0)}, drops: {s.get('drops', 0)}, "
                    f"push_errors: {s.get('push_errors', 0)}"
                )
            except Exception as e:
                logger.warning(f"[W2WSend] failed to report stats: {type(e).__name__}")

    def shutdown(self):
        """Gracefully terminate the server, process-safe.

        Please note that terminating server otherwise (e.g. by killing processes) may result in zombie processes.
        If you did already cause a zombie outbreak, your only option is to kill them with -9 (SIGKILL).
        """
        with self._shutdown_lock:
            if self._shutdown_started.is_set():
                return
            self._shutdown_started.set()
        self.ready.clear()
        self._stop_serving.set()  # release the reducer keep-alive in run() (no-op for normal servers)

        self.training_coordinator.shutdown()

        for handler in self.conn_handlers:
            handler.shutdown()
        logger.debug("Connection handlers terminated")

        if self.activation_cache is not None:
            self.activation_cache.close()

        if self.w2w_origin_injector is not None:
            self.w2w_origin_injector.shutdown()

        if self.w2w_membership_feeder is not None:
            self.w2w_membership_feeder.stop()

        if self.w2w_coordinator is not None:
            self.w2w_coordinator.close()

        if self.w2w_forward_driver is not None:
            self.w2w_forward_driver.close()

        if self._w2w_manager is not None:
            self._w2w_manager.shutdown()

        if self.module_backends:
            self.dht_handler_thread.stop.set()
            self.dht_handler_thread.join()

        if self.checkpoint_saver is not None and isinstance(self.checkpoint_saver, threading.Thread):
            self.checkpoint_saver.stop.set()
            self.checkpoint_saver.join()

        self.dht.shutdown()

        if not self.auxiliary:
            logger.debug("Shutting down runtime")
            self.runtime.shutdown()

        logger.info("Server shutdown successfully")


def _generate_uids(
    num_experts: int,
    expert_pattern: str | None,
    dht: DHT | None = None,
    attempts_per_expert: int = 10,
) -> list[str]:
    """Sample experts from a given pattern, remove duplicates.

    Args:
        num_experts (int): Sample this many unique expert uids.
        expert_pattern (str | None): A string pattern or a list of expert uids, example: myprefix.[0:32].[0:256] means "sample random experts between myprefix.0.0 and myprefix.255.255".
        dht (DHT | None, optional): If specified, uses this DHT to check that expert uids are not yet occupied by other peers. Defaults to None.
        attempts_per_expert (int, optional): Give up if unable to generate a new expert uid after this many attempts per uid. Defaults to 10.

    Returns:
        list[str]: List of generated unique expert uids.

    **Note:**
        This method is not strictly process-safe. If several servers run it concurrently, they have a small chance of sampling duplicate expert uids.
    """
    remaining_attempts = attempts_per_expert * num_experts
    found_uids, attempted_uids = list(), set()

    def _generate_uid():
        if expert_pattern is None:
            return f"expert{UID_DELIMITER}{attempts_per_expert * num_experts - remaining_attempts}"

        uid = []
        for block in expert_pattern.split(UID_DELIMITER):
            try:
                if "[" not in block and "]" not in block:
                    uid.append(block)
                elif block.startswith("[") and block.endswith("]") and ":" in block:
                    slice_start, slice_end = map(int, block[1:-1].split(":"))
                    uid.append(str(random.randint(slice_start, slice_end - 1)))
                else:
                    raise ValueError("Block must be either fixed or a range [from:to]")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                raise ValueError(f"Expert pattern {expert_pattern} has invalid block {block}, {e}") from e
        return UID_DELIMITER.join(uid)

    while remaining_attempts > 0 and len(found_uids) < num_experts:
        # 1. sample new expert uids at random
        new_uids = []
        while len(new_uids) + len(found_uids) < num_experts and remaining_attempts > 0:
            new_uid = _generate_uid()
            remaining_attempts -= 1
            if new_uid not in attempted_uids:
                attempted_uids.add(new_uid)
                new_uids.append(new_uid)

        # 2. look into DHT (if given) and remove duplicates
        if dht is not None:
            existing_expert_uids = {
                found_expert.uid for found_expert in get_experts(dht, new_uids) if found_expert is not None
            }
            new_uids = [new_uid for new_uid in new_uids if new_uid not in existing_expert_uids]

        found_uids += new_uids

    if len(found_uids) != num_experts:
        logger.warning(
            f"Found only {len(found_uids)} out of {num_experts} free expert uids after "
            f"{attempts_per_expert * num_experts} attempts"
        )
    return found_uids
