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

from __future__ import annotations

import copy
import logging
import re

from collections.abc import Callable

from agora_server.hivemind.utils.logging import TextStyle, use_colors
from agora_server.prometheus.roles import PUSH_FAILURE_ALERT_MESSAGE


# ---------------------------------------------------------------------------
# Extra colours
# ---------------------------------------------------------------------------


def _c(code: str) -> str:
    return f"\033[{code}m" if use_colors else ""


_GREEN = _c("32")
_CYAN = _c("36")
_YELLOW = _c("93")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _humanize_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# ---------------------------------------------------------------------------
# Transform functions
# ---------------------------------------------------------------------------

_RE_DHT = re.compile(r"Running DHT node on \['/ip4/(?P<ip>[^/]+)/tcp/(?P<port>\d+)/p2p/(?P<peer>[^']+)'\]")


def _transform_dht(m: re.Match, _msg: str) -> list[tuple[str, str]]:
    return [
        ("NETWORK", "Connected to Agora"),
        ("NETWORK", f"PeerID: {m.group('peer')}"),
        ("NETWORK", f"Address: {m.group('ip')}:{m.group('port')}"),
    ]


_RE_INITIALIZING = re.compile(r"Initializing expert")


def _transform_initializing(_m: re.Match, _msg: str) -> list[tuple[str, str]]:
    return [("SERVER", "Initializing expert, this may take a few minutes...")]


_RE_SERVER_STARTED = re.compile(r"Server started with (\d+) modules?:")


def _transform_server_started(_m: re.Match, _msg: str) -> list[tuple[str, str]]:
    sep = f"{TextStyle.BOLD}{'─' * 60}{TextStyle.RESET}"
    return [
        ("", sep),
        ("SERVER", "Training started"),
    ]


_RE_MODULE_DETAIL = re.compile(r"^(?P<name>[^:]+):\s+\w+,\s+(?P<params>[\d,]+)\s+parameters$")


def _transform_module_detail(m: re.Match, _msg: str) -> list[tuple[str, str]]:
    name = m.group("name")
    params = int(m.group("params").replace(",", ""))
    stage = name.split(".")[0]
    return [
        (
            "SERVER",
            f"Expert {TextStyle.BOLD}{name}{TextStyle.RESET} ({_humanize_params(params)} params, {stage} stage)",
        )
    ]


_RE_STEP = re.compile(r"Transitioning to epoch (\d+)")


def _transform_step(m: re.Match, _msg: str) -> list[tuple[str, str]]:
    n = m.group(1)
    return [("TRAINING", f"Training step {TextStyle.BOLD}{n}{TextStyle.RESET}")]


_RE_AR_FINISHED = re.compile(r"All-reduce round finished at local epoch #(\d+)")


def _transform_ar_finished(m: re.Match, _msg: str) -> list[tuple[str, str]]:
    n = m.group(1)
    return [("TRAINING", f"Step {TextStyle.BOLD}{n}{TextStyle.RESET} averaged")]


_RE_PROCESSED = re.compile(r"^Processed (\d+) batches in last (\d+) seconds:$")


def _transform_processed(m: re.Match, _msg: str) -> list[tuple[str, str]]:
    count = m.group(1)
    return [("PROGRESS", f"Processed {TextStyle.BOLD}{count}{TextStyle.RESET} batches in the last {m.group(2)}s")]


_RE_POOL_DETAIL = re.compile(
    r"^(\S+?)_(forward|backward):\s+"
    r"(\d+)\s+batches\s+\(([^)]+)\),.*"
)

_DIRECTION_LABEL = {"forward": "Forward pass", "backward": "Backward pass"}


def _transform_pool_detail(m: re.Match, _msg: str) -> list[tuple[str, str]]:
    direction = m.group(2)
    batches = m.group(3)
    label = _DIRECTION_LABEL.get(direction, direction.capitalize())
    return [("PROGRESS", f"  {label}: {batches} batches")]


_RE_SHUTDOWN = re.compile(r"^Shutting down Agora\.\.\.$")


def _transform_shutdown(_m: re.Match, _msg: str) -> list[tuple[str, str]]:
    return [("SERVER", "Shutting down... Goodbye!")]


_RE_DOWNLOAD_START = re.compile(r"Starting state download with (\d+) threads")


def _transform_download_start(m: re.Match, _msg: str) -> list[tuple[str, str]]:
    return [("DOWNLOAD", "Downloading model weights...")]


_RE_DOWNLOAD_PROGRESS = re.compile(r"State download progress: ([\d.]+)%, speed: ([\d.]+) MB/s, ETA: (\S+)")


def _transform_download_progress(m: re.Match, _msg: str) -> list[tuple[str, str]]:
    pct, speed, eta = m.group(1), m.group(2), m.group(3)
    return [("DOWNLOAD", f"{TextStyle.BOLD}{pct}%{TextStyle.RESET} - {speed} MB/s, ETA: {eta}")]


_RE_DOWNLOAD_SUCCESS = re.compile(r"State \((\d+) MB\) downloaded in ([\d.]+) sec")


def _transform_download_success(m: re.Match, _msg: str) -> list[tuple[str, str]]:
    size, secs = m.group(1), m.group(2)
    return [
        (
            "DOWNLOAD",
            f"Model weights downloaded ({TextStyle.BOLD}{size} MB{TextStyle.RESET} in {secs}s). Waiting for authorization...",
        )
    ]


_RE_AUTH_QUEUE = re.compile(r"Authorization queue\. Position: (\d+), Estimated wait: (.+)$")


def _transform_auth_queue(m: re.Match, _msg: str) -> list[tuple[str, str]]:
    pos, wait = m.group(1), m.group(2)
    return [
        (
            "AUTH",
            f"Authorization queue: position {TextStyle.BOLD}{pos}{TextStyle.RESET}, estimated wait: {TextStyle.BOLD}{wait}{TextStyle.RESET}",
        )
    ]


_RE_AUTH_WAITLIST = re.compile(r"On waitlist\. Position: (\d+)\.")


def _transform_auth_waitlist(m: re.Match, _msg: str) -> list[tuple[str, str]]:
    pos = m.group(1)
    return [
        (
            "AUTH",
            f"On waitlist: position {TextStyle.BOLD}{pos}{TextStyle.RESET}. "
            "Waiting for a slot to open, you will join automatically when capacity is available.",
        )
    ]


_RE_AUTH_GRANTED = re.compile(r"Access for user (\S+) has been granted until (.+) UTC")


def _transform_auth_granted(m: re.Match, _msg: str) -> list[tuple[str, str]]:
    user = m.group(1)
    return [("AUTH", f"Access granted for {TextStyle.BOLD}{user}{TextStyle.RESET}")]


_RE_NODE_STARTING = re.compile(r"Agora starting\. Verbose logs are written to: (.+)")


def _transform_node_starting(m: re.Match, _msg: str) -> list[tuple[str, str]]:
    fname = m.group(1)
    return [("SERVER", f"Starting up. Verbose logs are written to: {TextStyle.BOLD}{fname}{TextStyle.RESET}")]


_RE_NODE_NAME = re.compile(r"^Node name: (\S+)")


def _transform_node_name(m: re.Match, _msg: str) -> list[tuple[str, str]]:
    name = m.group(1)
    return [("SERVER", f"Node name: {TextStyle.BOLD}{name}{TextStyle.RESET} (look for this in the dashboard)")]


_RE_SPEED_TEST_START = re.compile(r"Testing internet speed\.\.\.")


def _transform_speed_test_start(_m: re.Match, _msg: str) -> list[tuple[str, str]]:
    return [("NETWORK", "Running internet speed test...")]


_RE_DOWNLOAD_SPEED = re.compile(r"Download Speed: (?P<speed>[\d.]+) Mbps")


def _transform_download_speed(m: re.Match, _msg: str) -> list[tuple[str, str]]:
    return [("NETWORK", f"Download: {m.group('speed')} Mbps")]


_RE_UPLOAD_SPEED = re.compile(r"Upload Speed: (?P<speed>[\d.]+) Mbps")


def _transform_upload_speed(m: re.Match, _msg: str) -> list[tuple[str, str]]:
    return [("NETWORK", f"Upload: {m.group('speed')} Mbps")]


_RE_LATENCY = re.compile(r"Latency: (?P<latency>[\d.]+) ms")


def _transform_latency(m: re.Match, _msg: str) -> list[tuple[str, str]]:
    return [("NETWORK", f"Latency: {m.group('latency')} ms")]


# ---------------------------------------------------------------------------
# Ghost-phase patterns
# ---------------------------------------------------------------------------

_RE_GHOST_EXIT = re.compile(r"Exiting ghost mode at \d+ local epoch")
_RE_GHOST_PHASE1_DURATION = re.compile(r"Ghost phase 1 will last (\d+) steps, ending at local epoch (\d+)")
_RE_GHOST_PHASE2_DURATION = re.compile(r"Ghost phase 2 will last (\d+) steps, ending at local epoch (\d+)")


# ---------------------------------------------------------------------------
# Pattern registry
# ---------------------------------------------------------------------------

_CLI_PATTERNS: list[tuple[str, re.Pattern, Callable | None]] = [
    # Startup
    ("SERVER", _RE_NODE_STARTING, _transform_node_starting),
    ("SERVER", _RE_NODE_NAME, _transform_node_name),
    # Download lifecycle
    ("DOWNLOAD", _RE_DOWNLOAD_START, _transform_download_start),
    ("DOWNLOAD", _RE_DOWNLOAD_PROGRESS, _transform_download_progress),
    ("DOWNLOAD", _RE_DOWNLOAD_SUCCESS, _transform_download_success),
    # Authorization
    ("AUTH", _RE_AUTH_WAITLIST, _transform_auth_waitlist),
    ("AUTH", _RE_AUTH_QUEUE, _transform_auth_queue),
    ("AUTH", _RE_AUTH_GRANTED, _transform_auth_granted),
    # Network / server / training patterns
    ("NETWORK", _RE_SPEED_TEST_START, _transform_speed_test_start),
    ("NETWORK", _RE_DOWNLOAD_SPEED, _transform_download_speed),
    ("NETWORK", _RE_UPLOAD_SPEED, _transform_upload_speed),
    ("NETWORK", _RE_LATENCY, _transform_latency),
    ("NETWORK", _RE_DHT, _transform_dht),
    ("SERVER", _RE_INITIALIZING, _transform_initializing),
    ("SERVER", _RE_SERVER_STARTED, _transform_server_started),
    ("SERVER", _RE_MODULE_DETAIL, _transform_module_detail),
    ("TRAINING", _RE_STEP, _transform_step),
    ("TRAINING", _RE_AR_FINISHED, _transform_ar_finished),
    ("PROGRESS", _RE_PROCESSED, _transform_processed),
    ("PROGRESS", _RE_POOL_DETAIL, _transform_pool_detail),
    ("SERVER", _RE_SHUTDOWN, _transform_shutdown),
]


# ---------------------------------------------------------------------------
# Tag colours
# ---------------------------------------------------------------------------

_TAG_STYLE: dict[str, str] = {
    "DOWNLOAD": _GREEN,
    "AUTH": TextStyle.PURPLE,
    "NETWORK": _CYAN,
    "SERVER": TextStyle.BLUE,
    "TRAINING": _GREEN,
    "PROGRESS": _YELLOW,
    "SYNC": _CYAN,
    "WARNING": TextStyle.ORANGE,
    "ERROR": TextStyle.RED,
}


# ---------------------------------------------------------------------------
# Filter & Formatter
# ---------------------------------------------------------------------------


class CLILogFilter(logging.Filter):
    """Only pass log records that are important for the CLI user.

    Attaches ``cli_tag`` and ``cli_lines`` attributes to accepted records
    so the formatter can render friendly output.
    """

    def __init__(self, name: str = "") -> None:
        super().__init__(name)
        self._sync_phase: str | None = None

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            msg = record.getMessage()
            # Persistent metrics push failure: render as a separated block so it stands
            # out from scrolling logs
            if PUSH_FAILURE_ALERT_MESSAGE in msg:
                sep = f"{TextStyle.RED}{TextStyle.BOLD}{'─' * 60}{TextStyle.RESET}"
                record.cli_lines = [("", sep), *[("ERROR", line) for line in msg.splitlines()], ("", sep)]
                record.cli_tag = "ERROR"
                return True
            record.cli_tag = "ERROR" if record.levelno >= logging.ERROR else "WARNING"
            record.cli_lines = None
            return True

        msg = record.getMessage()

        # Ghost-phase state transitions
        m = _RE_GHOST_PHASE1_DURATION.search(msg)
        if m:
            self._sync_phase = "phase1"
            steps, end = m.group(1), m.group(2)
            record.cli_lines = [
                ("SYNC", "Synchronising weights with peers. Node won't process batches in this phase."),
                ("SYNC", f"This phase will last {steps} steps (until local epoch {end})."),
            ]
            record.cli_tag = "SYNC"
            return True

        m = _RE_GHOST_PHASE2_DURATION.search(msg)
        if m:
            self._sync_phase = "phase2"
            steps, end = m.group(1), m.group(2)
            record.cli_lines = [
                (
                    "SYNC",
                    "Synchronising optimizer state. Node is now processing batches, but doesn't contribute to weight averaging yet.",
                ),
                ("SYNC", f"This phase will last {steps} steps (until local epoch {end})."),
            ]
            record.cli_tag = "SYNC"
            return True

        if _RE_GHOST_EXIT.search(msg):
            if self._sync_phase is not None:
                self._sync_phase = None
                record.cli_lines = [("SYNC", "Sync complete. Node is now fully contributing to training.")]
                record.cli_tag = "SYNC"
                return True
            return False

        for tag, pattern, transform in _CLI_PATTERNS:
            m = pattern.search(msg)
            if m:
                # Suppress "Processed 0 batches" during phase-1 sync (no batches are processed)
                if self._sync_phase == "phase1" and pattern is _RE_PROCESSED and m.group(1) == "0":
                    return False

                if transform is not None:
                    lines = transform(m, msg)
                    record.cli_lines = lines
                    record.cli_tag = next((t for t, _ in lines if t), tag)
                else:
                    record.cli_lines = None
                    record.cli_tag = tag
                return True

        return False


class CLIFormatter(logging.Formatter):
    """Compact, colour-coded formatter for console output.

    When the filter attached ``cli_lines``, each (tag, message) pair is
    rendered as a separate visual line.  Otherwise the raw message is used.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Suppress tracebacks in CLI output
        if record.exc_info or record.exc_text:
            record = copy.copy(record)
            record.exc_info = None
            record.exc_text = None

        ts = self.formatTime(record, "%H:%M:%S")
        lines: list[tuple[str, str]] | None = getattr(record, "cli_lines", None)

        if lines:
            parts = []
            for tag, text in lines:
                if not tag:
                    parts.append(text)
                else:
                    colour = _TAG_STYLE.get(tag, TextStyle.BLUE)
                    prefix = f"{ts} {colour}{TextStyle.BOLD}[{tag}]{TextStyle.RESET}"
                    parts.append(f"{prefix} {text}")
            return "\n".join(parts)

        tag = getattr(record, "cli_tag", "INFO")
        colour = _TAG_STYLE.get(tag, TextStyle.BLUE)
        prefix = f"{ts} {colour}{TextStyle.BOLD}[{tag}]{TextStyle.RESET}"
        return f"{prefix} {record.getMessage()}"
