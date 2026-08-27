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
import threading

from collections import defaultdict
from itertools import chain
from queue import SimpleQueue
from selectors import EVENT_READ, DefaultSelector
from statistics import mean
from time import perf_counter
from typing import Any, NamedTuple

import torch

from prefetch_generator import BackgroundGenerator

from agora_server.hivemind.moe.server.module_backend import ModuleBackend
from agora_server.hivemind.moe.server.task_pool import TaskPoolBase
from agora_server.hivemind.utils import get_logger


BatchStats = NamedTuple("BatchStats", (("batch_size", int), ("processing_time", float), ("batch_await_time", float)))

logger = get_logger(__name__)


class Runtime(threading.Thread):
    """A group of processes that processes incoming requests for multiple module backends on a shared device.

    Runtime is usually created and managed by Server, humans need not apply.

    For debugging, you can start runtime manually with .start() or .run()

    >>> module_backends = {'expert_name': ModuleBackend(**kwargs)}
    >>> runtime = Runtime(module_backends)
    >>> runtime.start()  # start runtime in background thread. To start in current thread, use runtime.run()
    >>> runtime.ready.wait()  # await for runtime to load all experts on device and create request pools
    >>> future = runtime.module_backends['expert_name'].forward_pool.submit_task(*module_inputs)
    >>> print("Returned:", future.result())
    >>> runtime.shutdown()
    """

    SHUTDOWN_TRIGGER = "RUNTIME SHUTDOWN TRIGGERED"

    def __init__(
        self,
        module_backends: dict[str, ModuleBackend],
        prefetch_batches: int = 1,
        sender_threads: int = 1,
        device: torch.device | None = None,
        stats_report_interval: int | None = None,
        coalesce_batches: bool = True,
    ):
        """Initialize Runtime.

        Args:
            module_backends (dict[str, ModuleBackend]): A dict [expert uid -> ModuleBackend].
            prefetch_batches (int, optional): Form up to this many batches in advance. Defaults to 1. Use more if not using coalesce_batches.
            sender_threads (int, optional): Dispatches outputs from finished batches using this many asynchronous threads. Defaults to 1.
            device (torch.device | None, optional): If specified, moves all experts and data to this device via .to(device=device). If you want to manually specify devices for each expert (in their forward pass), leave device=None. Defaults to None.
            stats_report_interval (int | None, optional): Interval to collect and log statistics about runtime performance. Defaults to None.
        """
        super().__init__()
        self.module_backends = module_backends
        self.pools = tuple(chain(*(backend.get_pools() for backend in module_backends.values())))
        self.device, self.prefetch_batches, self.sender_threads = device, prefetch_batches, sender_threads
        self.coalesce_batches = coalesce_batches

        if coalesce_batches and prefetch_batches > 1:
            logger.warning(
                f"coalesce_batches=True overrides prefetch_batches={prefetch_batches} to 1. "
                "prefetch_batches>1 defeats coalescing by eagerly draining the pipe."
            )
        self.shutdown_recv, self.shutdown_send = mp.Pipe(duplex=False)
        self.shutdown_trigger = mp.Event()
        self.ready = mp.Event()  # event is set iff server is currently running and ready to accept batches

        self.stats_report_interval = stats_report_interval
        if self.stats_report_interval is not None:
            self.stats_reporter = StatsReporter(self.stats_report_interval)

    def run(self):
        for pool in self.pools:
            if not pool.is_alive():
                pool.start()
        if self.device is not None:
            for backend in self.module_backends.values():
                backend.module.to(self.device)

        with mp.pool.ThreadPool(self.sender_threads) as output_sender_pool:
            try:
                self.ready.set()
                if self.stats_report_interval is not None:
                    self.stats_reporter.start()
                logger.info("Started")

                batch_iterator = self.iterate_minibatches_from_pools()
                if self.coalesce_batches:
                    # Prefetch exactly 1 coalesced batch ahead for I/O overlap.
                    # Higher values defeat coalescing by eagerly draining the pipe.
                    batch_iterator = BackgroundGenerator(batch_iterator, max_prefetch=1)
                elif self.prefetch_batches > 0:
                    batch_iterator = BackgroundGenerator(batch_iterator, self.prefetch_batches)

                yield_start_time = perf_counter()
                for pool, batch_indices, batch, batch_sizes in batch_iterator:
                    batch_yield_time = perf_counter() - yield_start_time
                    logger.debug(f"Processing coalesced batch {batch_indices} from pool {pool.name}")
                    start = perf_counter()
                    try:
                        outputs, total_batch_size = self.process_batch(pool, batch_indices[0], *batch)
                        batch_processing_time = perf_counter() - start
                        logger.debug(
                            f"Pool {pool.name}: batch {batch_indices} processed, "
                            f"size {total_batch_size} ({len(batch_indices)} coalesced)"
                        )
                        if self.stats_report_interval is not None:
                            self.stats_reporter.report_stats(
                                pool.name, total_batch_size, batch_processing_time, batch_yield_time
                            )

                        self._send_coalesced_outputs(output_sender_pool, pool, batch_indices, batch_sizes, outputs)
                    except KeyboardInterrupt:
                        raise
                    except BaseException as exception:
                        logger.exception(f"Caught {exception}, attempting to recover")
                        for idx in batch_indices:
                            output_sender_pool.apply_async(pool.send_exception_from_runtime, args=[idx, exception])
                    yield_start_time = perf_counter()

            finally:
                if not self.shutdown_trigger.is_set():
                    self.shutdown()

    @staticmethod
    def _send_coalesced_outputs(output_sender_pool, pool, batch_indices, batch_sizes, outputs):
        """Split coalesced outputs back to original batches and dispatch to task futures."""
        if len(batch_indices) == 1:
            output_sender_pool.apply_async(pool.send_outputs_from_runtime, args=[batch_indices[0], outputs])
        else:
            splits = [torch.split_with_sizes(out, batch_sizes, dim=0) for out in outputs]
            for i, idx in enumerate(batch_indices):
                sub_outputs = tuple(s[i] for s in splits)
                output_sender_pool.apply_async(pool.send_outputs_from_runtime, args=[idx, sub_outputs])

    def process_batch(self, pool: TaskPoolBase, batch_index: int, *batch: torch.Tensor) -> tuple[Any, int]:
        """Process one batch of tasks from a given pool, return a batch of results and total batch size."""
        outputs = pool.process_func(*batch)
        return outputs, outputs[0].size(0)

    def shutdown(self):
        """Gracefully terminate a running runtime."""
        logger.info("Shutting down Runtime")
        self.ready.clear()

        if self.stats_report_interval is not None:
            self.stats_reporter.stop.set()
            self.stats_reporter.join()

        logger.debug("Terminating pools")
        for pool in self.pools:
            if pool.is_alive():
                pool.terminate()
                pool.join()
        logger.debug("Pools terminated")

        # trigger background thread to shutdown
        self.shutdown_send.send(self.SHUTDOWN_TRIGGER)
        self.shutdown_trigger.set()

    def iterate_minibatches_from_pools(self, timeout: float | None = None):
        """Iteratively select non-empty pool with highest priority, drain and coalesce available batches.

        Yields (pool, batch_indices, merged_tensors, batch_sizes) where batch_indices and batch_sizes
        are lists tracking the original sub-batches so outputs can be split back after processing.
        """
        with DefaultSelector() as selector:
            for pool in self.pools:
                selector.register(pool.batch_receiver, EVENT_READ, pool)
            selector.register(self.shutdown_recv, EVENT_READ, self.SHUTDOWN_TRIGGER)

            while True:
                logger.debug("Waiting for inputs from task pools")
                ready_fds = selector.select()
                ready_objects = {key.data for (key, events) in ready_fds}
                if self.SHUTDOWN_TRIGGER in ready_objects:
                    break

                pool = min(ready_objects, key=lambda pool: pool.priority)

                # Load the first batch (guaranteed available since selector signalled)
                idx, tensors = pool.load_batch_to_runtime(timeout, self.device)

                if not self.coalesce_batches:
                    yield pool, [idx], tensors, [tensors[0].shape[0]]
                    continue

                max_size = pool.max_batch_size
                batch_indices = [idx]
                batch_tensors_list = [tensors]
                total_size = tensors[0].shape[0]

                # Drain additional available batches up to max_batch_size.
                # Hold one batch aside if adding it would exceed the limit, and
                # yield it separately so max_batch_size is strictly respected.
                overflow_batch = None
                while total_size < max_size and pool.batch_receiver.poll():
                    next_idx, next_tensors = pool.load_batch_to_runtime(None, self.device)
                    next_size = next_tensors[0].shape[0]
                    if total_size + next_size > max_size:
                        overflow_batch = (next_idx, next_tensors)
                        break
                    total_size += next_size
                    batch_indices.append(next_idx)
                    batch_tensors_list.append(next_tensors)

                batch_sizes = [t[0].shape[0] for t in batch_tensors_list]

                if len(batch_tensors_list) == 1:
                    yield pool, batch_indices, batch_tensors_list[0], batch_sizes
                else:
                    merged = [
                        torch.cat([b[i] for b in batch_tensors_list], dim=0) for i in range(len(batch_tensors_list[0]))
                    ]
                    logger.debug(
                        f"Coalesced {len(batch_tensors_list)} batches from {pool.name}, total size {total_size}"
                    )
                    yield pool, batch_indices, merged, batch_sizes

                # Yield overflow batch separately (loaded from pipe but would exceed max_batch_size)
                if overflow_batch is not None:
                    ov_idx, ov_tensors = overflow_batch
                    yield pool, [ov_idx], ov_tensors, [ov_tensors[0].shape[0]]


class StatsReporter(threading.Thread):
    def __init__(self, report_interval: int):
        super().__init__()
        self.report_interval = report_interval
        self.stop = threading.Event()
        self.stats_queue = SimpleQueue()

    def run(self):
        while not self.stop.wait(self.report_interval):
            pool_batch_stats = defaultdict(list)
            while not self.stats_queue.empty():
                pool_uid, batch_stats = self.stats_queue.get()
                pool_batch_stats[pool_uid].append(batch_stats)

            total_processed_batches = sum(len(pool_stats) for pool_stats in pool_batch_stats.values())
            logger.info(f"Processed {total_processed_batches} batches in last {self.report_interval} seconds:")
            for pool_uid, pool_stats in pool_batch_stats.items():
                total_batches = len(pool_stats)
                total_examples = sum(batch_stats.batch_size for batch_stats in pool_stats)
                avg_batch_size = mean(batch_stats.batch_size for batch_stats in pool_stats)
                avg_batch_await_time = mean(batch_stats.batch_await_time for batch_stats in pool_stats)
                total_time = sum(batch_stats.processing_time for batch_stats in pool_stats)
                batches_to_time = total_batches / total_time
                batch_performance = f"{batches_to_time:.2f} " + ("batches/s" if batches_to_time > 1 else "s/batch")

                examples_to_time = total_examples / total_time
                example_performance = f"{examples_to_time:.2f} " + (
                    "examples/s" if examples_to_time > 1 else "s/example"
                )

                logger.info(
                    f"{pool_uid}: "
                    f"{total_batches} batches ({batch_performance}), "
                    f"{total_examples} examples ({example_performance}), "
                    f"avg batch size {avg_batch_size:.2f}, "
                    f"avg batch await time {avg_batch_await_time * 1000:.0f}ms"
                )

    def report_stats(self, pool_uid: str, batch_size: int, processing_time: float, batch_await_time: float):
        self.stats_queue.put_nowait((pool_uid, BatchStats(batch_size, processing_time, batch_await_time)))
