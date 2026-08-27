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

import argparse
import json
import logging
import os
import platform
import sys

from functools import partial
from logging.handlers import RotatingFileHandler
from pathlib import Path
from signal import SIGINT, signal, strsignal
from threading import Event, Thread
from typing import Any

import torch

from omegaconf import OmegaConf

from agora.security.authorizer import AgoraAuthorizer
from agora.utils import (
    clean_tmp,
    get_batch_size,
    infer_expert_params,
)
from agora.utils.cli_logger import CLIFormatter, CLILogFilter
from agora.utils.state_loader import StateDownloader, StateLoader
from agora_server.core.server.server import Server
from agora_server.hivemind.proto.runtime_pb2 import CompressionType
from agora_server.hivemind.utils.logging import get_logger
from agora_server.logging.log_monitor import LogMonitor, MonitorRule
from agora_server.logging.logger import PluralisLogger
from agora_server.models.arguments_registry import ModelArgumentsRegistry
from agora_server.monitor.gpu_util_monitor import GpuUtilizationTracker
from agora_server.monitor.mem_monitor import MemoryTracker
from agora_server.monitor.network_throughput import NetworkMonitor
from agora_server.prometheus.monitor import WorkerPromMonitor
from agora_server.security.auth import apply_shard_assignment, authorize_with_pluralis
from agora_server.security.validator_builder import ValidatorBuilder
from agora_server.security.validator_schema import WorkerKillSignalSchema
from agora_server.types import ServerCreationError
from agora_server.utils.common import build_cls
from agora_server.utils.configs import load_config, validate_config
from agora_server.utils.dht_partition import update_initial_peers
from agora_server.utils.node_info import get_node_info


logger = get_logger(__name__)

if platform.system().lower() == "darwin":
    print("macOS is not supported")
    sys.exit(1)


def parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    # Connection arguments
    parser.add_argument(
        "--host_maddrs",
        type=str,
        help="Multiaddrs to listen for external connections from other p2p instances",
    )
    parser.add_argument(
        "--announce_maddrs",
        type=str,
        help="Visible multiaddrs the host announces for external connections from other p2p instances",
    )
    parser.add_argument(
        "--initial_peers",
        type=str,
        nargs="*",
        required=True,
        default=[],
        help="Multiaddrs of one or more active DHT peers",
    )

    # Experiment arguments
    parser.add_argument("--config", type=str, required=True, help="Run configuration file.")

    # Authorization parameters
    parser.add_argument(
        "--identity_path",
        type=str,
        default="private.key",
        help="Path to identity file to be used in P2P",
    )
    parser.add_argument("--token", type=str, required=True, help="HuggingFace token")
    parser.add_argument("--email", type=str, default="", help="Email address")
    parser.add_argument("--auth_server", type=str, required=True, help="Authentication server URL")
    parser.add_argument("--prom_gateway", type=str, required=False, help="Prometheus gateway URL")
    parser.add_argument("--log_file", type=str, required=False, help="Log file location")
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to avoid download folder collisions when running on multiple GPUs.",
    )
    parser.add_argument(
        "--batch_size_override",
        type=int,
        default=None,
        help="Override the batch size derived from GPU memory in the run config.",
    )
    parser.add_argument(
        "--pithos_shard_index",
        type=int,
        default=None,
        help="Explicit corpus shard index for trainerless shard mode; required on head stages "
        "when the run config sets trainerless_data_mode=shard.",
    )

    args = parser.parse_args()
    args = vars(args)

    # Read run config file
    config_path = Path(args.pop("config"))
    agora_dir = Path(__file__).resolve().parent
    config_path = agora_dir / "configs" / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    # Load config file and update args
    config = load_config(config_path)
    validate_config(config)

    config_dict = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config_dict, dict):
        raise ValueError(f"Config at {config_path} can't be converted to a dictionary.")

    config_dict.update({k: v for k, v in args.items() if v is not None})
    return config_dict


def main():
    if not torch.backends.openmp.is_available():
        # Necessary to prevent the server from freezing after forks
        torch.set_num_threads(1)

    if hasattr(sys, "gettrace") and sys.gettrace() is not None:
        logger.error("Debug mode is not supported")
        sys.exit(1)

    # Parse arguments
    args = parse_args()

    # Set up logging
    log_level = args.pop("log_level", "info").upper()
    log_file = args.pop("log_file", "logs/server.log")
    agora_logger = PluralisLogger(log_level=log_level, log_file=log_file, multiprocess_logging=True)

    # Replace the console handler's formatter/filter
    cli_filter = CLILogFilter()
    cli_formatter = CLIFormatter()
    for handler in agora_logger.root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, RotatingFileHandler):
            handler.addFilter(cli_filter)
            handler.setFormatter(cli_formatter)

    log_file_abs = os.path.abspath(log_file)
    logger.info(f"Agora starting. Verbose logs are written to: {log_file_abs}")

    server = None
    log_monitor = None
    prom_monitor = None
    authorizer = None
    state_downloader = None
    exit_code = 0
    user_token = args.pop("token")
    user_email = args.pop("email")

    logger.info(f"Running with configuration: {json.dumps(args, indent=4, default=str)}")

    try:
        # Monitor log messages
        experiment_prefix = args.pop("experiment_prefix", "pluralis")
        raw_rules = args.pop("log_monitor_rules", None)
        rules = [MonitorRule.from_dict(r) for r in raw_rules] if raw_rules else None
        log_monitor = LogMonitor(
            active_period_timeout=args.pop("active_period_timeout"),
            experiment_prefix=experiment_prefix,
            rules=rules,
            kill_message_check_interval=60,
        )
        agora_logger.add_handler(log_monitor.queue_handler)
        log_monitor.start()

        # Check pytorch version
        if not torch.__version__.startswith("2.11"):
            logger.error("Wrong pytorch version. Please install torch 2.11")
            sys.exit(1)

        # Clean tmp folder
        gpu_id = args.pop("gpu_id", 0)
        clean_tmp(gpu_id=gpu_id)

        # Collect information about the node
        node_info = get_node_info(args["initial_peers"], do_speed_test=True, find_best_speed_server=True)

        if node_info.latency is None:
            logger.error("Latency measurement failed. Exiting run.")
            sys.exit(1)

        allow_cpu = args.pop("allow_cpu", False)
        if node_info.device_name == "cpu" and not allow_cpu:
            logger.error("CUDA is not available. Exiting run.")
            sys.exit(1)

        # Select batch size
        batch_size_override = args.pop("batch_size_override", None)
        max_batch_size_config = args.pop("max_batch_size")
        recommended_batch_size = get_batch_size(node_info.gpu_memory, max_batch_size_config)
        if batch_size_override is not None:
            args["max_batch_size"] = min(batch_size_override, recommended_batch_size)
        else:
            args["max_batch_size"] = recommended_batch_size
        logger.info(f"Selected batch size: {args['max_batch_size']}")

        # Authorize
        check_integrity = args.pop("check_integrity", True)
        check_port = args.pop("check_port", True)
        prom_gateway = args.pop("prom_gateway", "https://metrics.pluralis.ai")

        authorizer = authorize_with_pluralis(
            node_info=node_info,
            user_token=user_token,
            user_email=user_email,
            role="worker",
            auth_server=args.pop("auth_server"),
            identity_path=args["identity_path"],
            announce_maddrs=args["announce_maddrs"],
            host_port=int(args["host_maddrs"].split("/")[4]),
            check_integrity=check_integrity,
            check_port=check_port,
            authorizer_cls=AgoraAuthorizer,
            hostname="agora",
        )

        state_downloader = None
        if not args.pop("load_state_from_peer", False):
            # Wait until the node is past the waitlist
            authorizer.wait_until_download_allowed(timeout=None)
            state_downloader = StateDownloader(authorizer.checkpoint_path, gpu_id=gpu_id).start()
            authorizer.set_state_downloader(state_downloader)

        auth_exc: list[BaseException] = []

        def _run_auth() -> None:
            try:
                authorizer.wait_for_authorization(timeout=None)
            except BaseException as e:
                auth_exc.append(e)
                if state_downloader:
                    state_downloader.cancel()

        auth_thread = Thread(target=_run_auth, daemon=True, name="auth-wait")
        auth_thread.start()

        if state_downloader:
            try:
                state_downloader.ensure_downloaded()
            except RuntimeError:
                auth_thread.join(timeout=5)
                if auth_exc:
                    raise auth_exc[0] from None
                raise

        auth_thread.join()
        if auth_exc:
            raise auth_exc[0]

        pipeline_stage = str(authorizer.pipeline_stage)
        ip_address = args["announce_maddrs"].split("/")[2]
        args["host_maddrs"] = [args["host_maddrs"]]
        args["announce_maddrs"] = [args["announce_maddrs"]]

        expert_params = infer_expert_params(pipeline_stage, str(authorizer.expert_uid))
        args.update(expert_params)
        apply_shard_assignment(args, authorizer)

        # Add validators
        validator_builder = ValidatorBuilder()
        validators, _ = (
            validator_builder.add_signature_validator()
            .add_schema(WorkerKillSignalSchema, prefix=f"{experiment_prefix}_{authorizer.peer_id}")
            .build()
        )

        # Add expert parameters to args
        num_worker_dhts = args.pop("num_dht") - 1
        args["initial_peers"] = update_initial_peers(
            args["initial_peers"], pipeline_stage, args["num_stages"], num_dht=num_worker_dhts
        )

        # Add authorization details to log monitor
        log_monitor.add_auth_info(
            peer_id=authorizer.peer_id,
            monitor_public_key=authorizer.monitor_public_key,
        )

        # Prom logging
        if prom_gateway and prom_gateway.lower() == "none":
            prom_gateway = None

        prom_callback = None
        stats_report_interval = args.pop("stats_report_interval", 60)
        _, role_stage, role_stage_idx, role_peer_id, role_ts = authorizer.exported_role.split("-")
        node_name = f"{role_stage}-{role_stage_idx}-{role_peer_id[-6:]}-{role_ts[-4:]}"
        logger.info(f"Node name: {node_name} (look for this in the dashboard)")
        if prom_gateway:
            prom_monitor = WorkerPromMonitor(
                role_name="agora",
                role_raw="agora",
                exported_role=authorizer.exported_role,
                gateway_url=prom_gateway,
                instance=ip_address,
                stats_report_interval=stats_report_interval,
                auth_token=authorizer.pushgateway_auth_token,
                local_dump_dir=str(Path(log_file).parent),
            )
            agora_logger.add_handler(prom_monitor.queue_handler)
            prom_monitor.start()
            prom_callback = prom_monitor.start_logging

        # Build model arguments
        custom_args = args.pop("custom_args_path", None)
        if custom_args:
            ModelArgumentsRegistry.add_custom_arguments(custom_args)

        model_args = ModelArgumentsRegistry.get(
            args["model_name"],
            args.pop("model_config"),
            ss_component_path=args.pop("ss_component_path", None),
            stage=args.pop("stage"),
            expert_name=args["expert_name"],
        )

        # Build torch optimizer and training coordinator
        optim_cls = build_cls(**args.pop("optim_config"), partial_init=True)
        training_coordinator = build_cls(**args.pop("training_coordinator_config"), partial_init=True)

        # Set BW for the peer
        default_bandwidth = args.pop("bandwidth")
        if node_info.upload_speed:
            bandwidth = node_info.upload_speed
        else:
            bandwidth = default_bandwidth

        # Build gradient buffer
        grad_buffer_config = args.pop("grad_buffer_config", None)

        if grad_buffer_config:
            grad_buffer_factory = build_cls(**grad_buffer_config, partial_init=True)
        else:
            grad_buffer_factory = None

        # Build state averager
        state_avg_config = args.pop("state_avg_config")
        state_avg_config["init_args"]["compression"] = build_cls(state_avg_config["init_args"].pop("compression"))
        state_avg_config["init_args"]["state_compression"] = build_cls(
            state_avg_config["init_args"].pop("state_compression")
        )
        state_avg_factory = build_cls(**state_avg_config, partial_init=True)

        # Compression
        compression_type = args.pop("compression")
        compression = getattr(CompressionType, compression_type)

        # Select device
        if node_info.device_name == "cpu":
            device = "cpu"
        elif node_info.device_name.startswith("mps"):
            device = "mps"
        else:
            device = "cuda"

        logger.info(f"Using {device} device")

        # Log machine usage
        NetworkMonitor(log_per_peer_tcp=args.pop("log_per_peer_tcp", True))
        MemoryTracker()
        if torch.cuda.is_available():
            GpuUtilizationTracker()

        # Start server
        server = Server.create(
            model_conf=model_args,
            experiment_prefix=experiment_prefix,
            training_coordinator_factory=training_coordinator,
            torch_optim_factory=optim_cls,
            state_avg_factory=state_avg_factory,
            grad_buffer_factory=grad_buffer_factory,
            start=True,
            compression=compression,
            record_validators=validators,
            authorizer=authorizer,
            log_monitor=log_monitor,
            prom_monitor_callback=prom_callback,
            bandwidth=bandwidth,
            device=device,
            stats_report_interval=stats_report_interval,
            load_checkpoint=(state_downloader is not None),
            checkpoint_manager_factory=partial(
                StateLoader,
                local_path=state_downloader.get_dest(),
            )
            if state_downloader is not None
            else None,
            use_relay=False,
            **args,
        )

        exit_event = Event()

        def signal_handler(signal_number: int, _) -> None:
            logger.info(f"Caught signal {signal_number} ({strsignal(signal_number)})")
            exit_event.set()

        signal(SIGINT, signal_handler)

        while not exit_event.wait(timeout=1.0):
            if server.fatal_error is not None:
                raise RuntimeError("Server stopped after a terminal asynchronous failure") from server.fatal_error

    except ServerCreationError as e:
        exit_code = 1
        logger.error(f"Server failed during startup: {e}")
    except Exception as e:
        exit_code = 1
        logger.error(f"Caught exception in main: {e}")
    finally:
        logger.info("Shutting down Agora...")

        if state_downloader:
            logger.info("Cancelling state download...")
            state_downloader.cancel()

        if authorizer:
            logger.info("Shutting down authorizer...")
            authorizer.begin_shutdown()
            authorizer.cancel_queue_wait()

        if server:
            logger.info("Shutting down server...")
            server.shutdown()
            server.join()

        if log_monitor:
            logger.info("Shutting down log monitor...")
            agora_logger.root_logger.removeHandler(log_monitor.queue_handler)
            log_monitor.stop()

        if prom_monitor:
            logger.info("Shutting down prometheus...")
            agora_logger.root_logger.removeHandler(prom_monitor.queue_handler)
            prom_monitor.stop()

        logger.info("Shutting down logger...")
        agora_logger.stop()

        # Force exit
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
