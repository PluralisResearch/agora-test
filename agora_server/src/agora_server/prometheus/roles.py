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

import threading

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest, push_to_gateway, start_http_server
from prometheus_client.exposition import default_handler

from agora_server.hivemind.utils.logging import get_logger


logger = get_logger(__name__)

# Message contract: every push-failure line contains the phrase "push metrics" on its
# first line (exempted from the connection_error log monitor rule), and only persistent-
# failure alerts carry this sentinel (rendered as a high-visibility block by the CLI).
PUSH_FAILURE_ALERT_MESSAGE = "Unable to report training progress"

# Label under which push failures are counted in the role's error counter.
PUSH_FAILURE_ERROR_TYPE = "metrics_push"


@dataclass
class DirectMetricsConfig:
    """Configuration for direct metrics scraping."""

    port: int = 8000


@dataclass
class PushGatewayConfig:
    """Configuration for push gateway metrics."""

    gateway_url: str
    instance: str
    job_name: str = "pushgateway"
    push_period: float = 15
    auth_token: str | None = None
    # Warn on the first failure of a streak and on every Nth after (~5 min at the 15s period).
    warn_every_n_failures: int = 20
    # Streak length at which persistent-failure alerts begin (~2 h at the 15s period). None disables them.
    alert_after_n_failures: int | None = 480
    # Once alerts are active, repeat them every Nth failure (~15 min at the 15s period).
    alert_every_n_failures: int = 60
    # Directory for local metrics snapshots (written on shutdown and with each alert). None disables them.
    local_dump_dir: str | None = None


class BaseRoleMetrics:
    """Base metrics collection for any role in the distributed system."""

    def __init__(
        self,
        role_name: str,
        metrics_role: Literal["worker", "train"],
        role_raw: str = "",
        exported_role: str | None = None,
        direct_config: DirectMetricsConfig | None = None,
        push_config: PushGatewayConfig | None = None,
        start: bool = True,
    ):
        """Initialize Prometheus metrics for a role.

        Args:
            role_name (str): Name of the role instance (e.g., "worker_1").
            role_raw (str): Raw role name (e.g., "worker", "trainer", "agora" - needed for Pushgateway). Defaults to "".
            metrics_role (Literal[&quot;worker&quot;, &quot;train&quot;]): Role category used in metric names.
            exported_role (str | None, optional): Role name to expose to external systems (e.g., Pushgateway grouping). If None, defaults to role_name. Defaults to None.
            direct_config (DirectMetricsConfig | None, optional): Optional configs for direct metric scraping. Defaults to None.
            push_config (PushGatewayConfig | None, optional): Optional configs for metrics pushing. Defaults to None.
            start (bool, optional): If True, start the HTTP server or background push thread immediately. Defaults to True.
        """
        if direct_config and push_config:
            raise ValueError("Cannot use both direct_config and push_config")
        if not direct_config and not push_config:
            raise ValueError("Must provide either direct_config or push_config")

        # Create a new registry for this instance
        self.registry = CollectorRegistry()

        # Initialize metrics with role label
        self.role_name = role_name if direct_config else role_raw  # Different format is expected in push gateway
        self.exported_role = exported_role or role_name
        self.metrics_role = metrics_role
        self.direct_config = direct_config
        self.push_config = push_config
        self._consecutive_push_failures = 0

        self._init_metrics()
        self._init_additional_metrics()
        logger.info(
            f"Initialized Prometheus metrics for {'direct scraping' if direct_config else 'push gateway'}. Role {self.exported_role}."
        )

        if start:
            self.start_server()

    def _push_handler(
        self,
        url: str,
        method: str,
        timeout: float | None,
        headers: list[tuple[str, str]],
        data: bytes,
    ) -> Callable[[], None]:
        """Handler that implements HTTP/HTTPS connections with Token Auth."""

        def handle():
            if self.push_config and self.push_config.auth_token:
                headers.append(("Authorization", f"Bearer {self.push_config.auth_token}"))
            default_handler(url, method, timeout, headers, data)()

        return handle

    def start_server(self):
        if self.push_config:
            # Start background push thread
            self._stop_event = threading.Event()
            self._push_thread = threading.Thread(target=self._background_push_worker, daemon=True)
            self._push_thread.start()
        elif self.direct_config:
            # Start metrics server
            start_http_server(self.direct_config.port, registry=self.registry)

    def shutdown(self):
        """Clean shutdown of the background thread."""
        if hasattr(self, "_stop_event") and self._stop_event is not None:
            self._stop_event.set()
            self._push_thread.join(timeout=5)

    def record_operation(self, operation_type, cnt):
        """Record an operation occurrence."""
        self.task_counter.labels(role=self.role_name, operation_type=operation_type).set(cnt)

    def record_error(self, error_type):
        """Record an error occurrence."""
        self.error_counter.labels(role=self.role_name, error_type=error_type).inc()

    def record_flop(self, delta):
        """Record a delta change for FLOP operation."""
        self.flop_counter.labels(role=self.role_name).inc(delta)

    def time_operation(self, operation_type, time_val):
        """Get a timer context for timing operations."""
        return self.task_duration.labels(role=self.role_name, operation_type=operation_type).set(time_val)

    def _background_push_worker(self):
        """Background worker that pushes metrics periodically; snapshots and flushes on shutdown."""
        if not self.push_config:
            return

        while not self._stop_event.wait(self.push_config.push_period):
            try:
                self._attempt_push(timeout=10)
            except Exception:
                # A failure here must not kill the push thread; the node would silently stop reporting
                logger.warning("Unexpected error while trying to push metrics", exc_info=True)

        self._dump_metrics_locally(reason="shutdown")
        # Final flush, skipped when the gateway is failing (streak > 2) so it doesn't stall
        # shutdown; the short timeout fits within shutdown()'s join window
        if self._consecutive_push_failures <= 2:
            try:
                self._push_once(timeout=3)
            except Exception:
                logger.info("Final metrics push on shutdown failed")

    def _push_once(self, timeout: float) -> None:
        """Push the current registry state to the gateway, raising on failure."""
        if not self.push_config:
            return

        push_to_gateway(
            self.push_config.gateway_url,
            job=self.push_config.job_name,
            registry=self.registry,
            grouping_key={
                "env": "production",
                "instance": self.push_config.instance,
                "exported_role": self.exported_role,
            },
            timeout=timeout,
            handler=self._push_handler,
        )

    def _attempt_push(self, timeout: float = 10) -> None:
        """Push once and account for the success or failure of the attempt.

        A failed push loses no data: the registry is in-process and cumulative, so the next
        successful push reports the up-to-date totals. Failures are counted, not retried.
        """
        try:
            self._push_once(timeout)
        except Exception as e:
            self._on_push_failure(e)
        else:
            self._on_push_success()

    def _on_push_success(self) -> None:
        if self._consecutive_push_failures:
            # WARNING so the recovery is visible on the CLI console, which only shows WARNING and above
            logger.warning(
                f"Metrics push recovered after {self._consecutive_push_failures} failed attempts; "
                "all accumulated progress has been reported"
            )
            self._consecutive_push_failures = 0

    def _on_push_failure(self, error: Exception) -> None:
        """Count the failure and log with escalation.

        WARNING on the first failure of a streak and every warn_every_n_failures after.
        From alert_after_n_failures on, the warnings are replaced by a repeating ERROR
        alert telling the operator to stop the node, keep the logs and contact the team.
        """
        if not self.push_config:
            return

        self._consecutive_push_failures += 1
        self.record_error(PUSH_FAILURE_ERROR_TYPE)

        n = self._consecutive_push_failures
        config = self.push_config
        minutes = int(n * config.push_period // 60)
        alert_after = config.alert_after_n_failures
        if alert_after and n >= alert_after:
            if (n - alert_after) % config.alert_every_n_failures == 0:
                # The operator may stop the node over this, so keep the snapshot fresh
                self._dump_metrics_locally(reason="push_failure_alert")
                logger.error(
                    f"Failed to push metrics for ~{minutes} min ({n} consecutive attempts; "
                    f"last error: {error}). {PUSH_FAILURE_ALERT_MESSAGE}.\n"
                    "Training progress is NOT being recorded and cannot be credited.\n"
                    "Please stop this instance, save all logs and any metrics snapshot "
                    "next to the log file, and contact the Pluralis team."
                )
        elif n == 1:
            logger.warning(f"Failed to push metrics: {error}")
        elif n % config.warn_every_n_failures == 0:
            logger.warning(
                f"Failed to push metrics for ~{minutes} min ({n} consecutive attempts): {error}. "
                "Training progress is not being reported. It will be reported automatically if the "
                "connection recovers, but will be lost if the node stops first; check the node's "
                "network connection."
            )

    def _dump_metrics_locally(self, reason: str) -> None:
        """Write the current registry state to a local file, best effort.

        Preserves a record of accumulated progress when the final push may never happen.
        The registry is cumulative, so each snapshot supersedes the previous one.
        """
        if not self.push_config or not self.push_config.local_dump_dir:
            return

        try:
            dump_dir = Path(self.push_config.local_dump_dir)
            dump_dir.mkdir(parents=True, exist_ok=True)
            dump_path = dump_dir / f"metrics_snapshot_{self.exported_role}.prom"
            header = (
                f"# Metrics snapshot ({reason}) at {datetime.now(timezone.utc).isoformat()}\n"
                f"# exported_role: {self.exported_role}\n"
                f"# consecutive_push_failures: {self._consecutive_push_failures}\n"
            )
            dump_path.write_bytes(header.encode() + generate_latest(self.registry))
            logger.info(f"Saved metrics snapshot to {dump_path}")
        except Exception:
            # Exception text in the message could match a terminating log monitor rule;
            # keep the details in the traceback
            logger.warning("Could not save local metrics snapshot", exc_info=True)

    def _init_metrics(self):
        """Initialize common metrics."""
        # Operation counters
        self.task_counter = Gauge(
            f"{self.metrics_role}_operations",
            "Total operations performed",
            ["role", "operation_type"],
            registry=self.registry,
        )

        # Operation timing
        self.task_duration = Gauge(
            f"{self.metrics_role}_duration",
            "Time spent processing operation",
            ["role", "operation_type"],
            registry=self.registry,
        )

        # Error counter
        self.error_counter = Counter(
            f"{self.metrics_role}_errors",
            "Total errors encountered",
            ["role", "error_type"],
            registry=self.registry,
        )

        # Flop counter
        self.flop_counter = Counter(
            f"{self.metrics_role}_flop",
            "Total FLOP processed",
            ["role"],
            registry=self.registry,
        )

    def _init_additional_metrics(self):
        """Initialize role-specific metrics. Override in subclasses if needed."""
        pass


class WorkerRoleMetrics(BaseRoleMetrics):
    """Metrics collection for worker roles."""

    def __init__(
        self,
        role_name: str,
        role_raw: str = "worker",
        exported_role: str | None = None,
        direct_config: DirectMetricsConfig | None = None,
        push_config: PushGatewayConfig | None = None,
        start: bool = True,
    ):
        super().__init__(
            metrics_role="worker",
            role_name=role_name,
            role_raw=role_raw,
            exported_role=exported_role,
            direct_config=direct_config,
            push_config=push_config,
            start=start,
        )
