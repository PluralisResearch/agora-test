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
import logging
import multiprocessing as mp
import os
import queue
import re
import signal

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from logging.handlers import QueueHandler
from threading import Event, Thread

from agora_server.hivemind.utils.logging import get_logger


logger = get_logger(__name__)


class MonitorAction(str, Enum):
    TERMINATE = "terminate"
    COUNTED_TERMINATE = "counted_terminate"
    WINDOWED_COUNTED_TERMINATE = "windowed_counted_terminate"
    THRESHOLD_TERMINATE = "threshold_terminate"
    TRACK_ACTIVITY = "track_activity"
    RESET_COUNTERS = "reset_counters"
    SET_STARTED = "set_started"


@dataclass
class MonitorRule:
    name: str
    patterns: list[str]
    action: MonitorAction
    message: str = ""
    signal_type: str = "SIGINT"
    threshold: float = 1
    compare: str = "above"
    resets_counters: list[str] = field(default_factory=list)
    window_size: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "MonitorRule":
        return cls(
            name=data["name"],
            patterns=data["patterns"],
            action=MonitorAction(data["action"]),
            message=data.get("message", ""),
            signal_type=data.get("signal_type", "SIGINT"),
            threshold=data.get("threshold", 1),
            compare=data.get("compare", "above"),
            resets_counters=data.get("resets_counters", []),
            window_size=data.get("window_size", 0),
        )


class BaseLogMonitor(Thread):
    def __init__(self):
        super().__init__()

        # Create a log handler
        self._log_queue = mp.Queue()
        self.queue_handler = QueueHandler(self._log_queue)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        self.queue_handler.setFormatter(formatter)
        self._stop_event = Event()

    def stop(self):
        """Signal the monitor thread to stop and wait for it to finish."""
        if self.is_alive():
            self._stop_event.set()
            self.join(timeout=5)


class LogMonitor(BaseLogMonitor):
    def __init__(
        self,
        rules: list[MonitorRule] | None = None,
        peer_id: str = "",
        experiment_prefix: str = "",
        active_period_timeout: float | None = None,
        kill_message_check_interval: int | None = None,
    ):
        super().__init__()
        self.dht = None
        self.experiment_prefix = experiment_prefix
        self.peer_id = peer_id
        self.last_active = None
        self.active_period_timeout = timedelta(seconds=active_period_timeout) if active_period_timeout else None
        self.kill_message_check_interval = kill_message_check_interval
        self._async_loop = None
        self._counters: dict[str, int] = {}
        self._windows: dict[str, deque] = {}
        self.started = False

        if rules:
            self._compiled_rules: list[tuple[list[re.Pattern], MonitorRule]] = [
                ([re.compile(p) for p in rule.patterns], rule) for rule in rules
            ]
            for rule in rules:
                if rule.action == MonitorAction.WINDOWED_COUNTED_TERMINATE:
                    self._windows[rule.name] = deque(maxlen=rule.window_size)
        else:
            self._compiled_rules = []

    def _resolve_signal(self, signal_type: str) -> signal.Signals:
        return getattr(signal, signal_type)

    def stop(self):
        """Signal the monitor and its async thread to stop."""
        self._stop_event.set()
        if self._async_loop is not None and self._async_loop.is_running():
            for task in asyncio.all_tasks(self._async_loop):
                self._async_loop.call_soon_threadsafe(task.cancel)
        if self.is_alive():
            self.join(timeout=5)

    def connect_dht(self, dht):
        """Connect to existing DHT."""
        self.dht = dht

    def add_auth_info(self, peer_id: str, monitor_public_key: str):
        """Update monitor with authorization details."""
        self.peer_id = peer_id
        self.monitor_public_key = monitor_public_key

    def _run_async_kill_message_check(self):
        self._async_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._async_loop)
        try:
            self._async_loop.run_until_complete(self._kill_message_check())
        except asyncio.CancelledError:
            pass
        finally:
            self._async_loop.close()

    async def _kill_message_check(self):
        while not self._stop_event.is_set():
            if self.dht:
                await self.possibly_get_kill_message()
            # Sleep in small increments to respond to stop_event quickly
            for _ in range(self.kill_message_check_interval):
                if self._stop_event.is_set():
                    return
                await asyncio.sleep(1)

    async def possibly_get_kill_message(self):
        """Check if DHT contains record with termination signal."""
        try:
            response = await self.dht.get(
                key=f"{self.experiment_prefix}_{self.peer_id}_terminate",
                latest=False,
                return_future=True,
            )

            if not response:
                return

            response = list(response.value.items())
            for value in response:
                subkey = value[0]

                if subkey == f"[owner:{self.monitor_public_key}]".encode():
                    logger.error("Node received termination signal. Exiting run.")
                    os.killpg(os.getpgrp(), signal.SIGTERM)
                    return

        except Exception as e:
            logger.error(f"Error checking for kill message: {e}. Exiting run.")
            os.killpg(os.getpgrp(), signal.SIGTERM)

    def run(self):
        """Run monitor in background to report worker's contribution."""
        if self.kill_message_check_interval is not None:
            async_thread = Thread(target=self._run_async_kill_message_check, daemon=True)
            async_thread.start()

        while not self._stop_event.is_set():
            try:
                log_entry = self._log_queue.get(timeout=1).getMessage()  # Wait for a log entry.

                # Check activity timeout
                if (
                    self.started
                    and self.last_active is not None
                    and self.active_period_timeout is not None
                    and (datetime.now(timezone.utc) - self.last_active) > self.active_period_timeout
                ):
                    logger.error("An unknown server error occurred. Exiting run.")
                    os.killpg(os.getpgrp(), signal.SIGINT)
                    return

                # Check if the log entry matches any of the rules
                for compiled_patterns, rule in self._compiled_rules:
                    match = None
                    for p in compiled_patterns:
                        match = p.search(log_entry)
                        if match:
                            break
                    if not match:
                        continue

                    if rule.action == MonitorAction.TRACK_ACTIVITY:
                        self.last_active = datetime.now(timezone.utc)
                        break
                    elif rule.action == MonitorAction.RESET_COUNTERS:
                        for counter_name in rule.resets_counters:
                            if counter_name in self._windows:
                                self._windows[counter_name].append(0)
                            else:
                                self._counters[counter_name] = 0
                        break
                    elif rule.action == MonitorAction.TERMINATE:
                        logger.error(rule.message)
                        os.killpg(os.getpgrp(), self._resolve_signal(rule.signal_type))
                        return
                    elif rule.action == MonitorAction.COUNTED_TERMINATE:
                        self._counters[rule.name] = self._counters.get(rule.name, 0) + 1
                        if self._counters[rule.name] >= rule.threshold:
                            logger.error(rule.message)
                            os.killpg(os.getpgrp(), self._resolve_signal(rule.signal_type))
                            return
                        break
                    elif rule.action == MonitorAction.WINDOWED_COUNTED_TERMINATE:
                        self._windows[rule.name].append(1)
                        if sum(self._windows[rule.name]) >= rule.threshold:
                            logger.error(rule.message)
                            os.killpg(os.getpgrp(), self._resolve_signal(rule.signal_type))
                            return
                        break
                    elif rule.action == MonitorAction.THRESHOLD_TERMINATE:
                        try:
                            value = float(match.group(1))
                        except (IndexError, ValueError):
                            break
                        exceeded = value >= rule.threshold if rule.compare == "above" else value <= rule.threshold
                        if exceeded:
                            logger.error(rule.message)
                            os.killpg(os.getpgrp(), self._resolve_signal(rule.signal_type))
                            return
                        break
                    elif rule.action == MonitorAction.SET_STARTED:
                        self.started = True
                        break

            except queue.Empty:
                continue
