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

import os
import re
import subprocess
import threading
import time

import psutil

from agora_server.hivemind.utils.logging import get_logger


logger = get_logger(__name__)


LOOPBACK_PREFIXES = ("127.", "[::1]")
MAX_PEER_TCP_LINES = 10


class NetworkMonitor(threading.Thread):
    """Monitor network throughput for a specific network interface or all interfaces."""

    def __init__(
        self,
        interface: str | None = None,
        interval: int = 15,
        duration: int | None = None,
        log_per_peer_tcp: bool = True,
    ):
        """Initialize the NetworkMonitor.

        Args:
            interface (str | None, optional): Specific network interface to monitor (e.g., 'eth0'). Defaults to None.
            interval (int, optional): Time between measurements in seconds. Defaults to 60.
            duration (int | None, optional): Total monitoring duration in seconds (None for continuous monitoring). Defaults to None.
            log_per_peer_tcp (bool, optional): Each interval, log a summary of remote TCP peers
                (rtt / retransmits / delivery rate, from `ss -ti`) plus a detail line for the worst
                few connections. Best effort. Defaults to True.
        """
        super().__init__(daemon=True)

        self.interface = interface
        self.interval = interval
        self.duration = duration

        self.log_per_peer_tcp = log_per_peer_tcp
        self._per_peer_tcp_disabled = False

        self.start()

    def get_tcp_connection_count(self) -> int:
        """Get the count of TCP connections from host system."""
        try:
            tcp_file = "/proc/1/root/proc/net/tcp"
            if os.path.exists(tcp_file):
                with open(tcp_file, "r") as f:
                    return len(f.readlines()) - 1  # Subtract 1 for header line
        except (OSError, PermissionError):
            pass

        # method fails, return -1 to indicate unavailable
        return -1

    @staticmethod
    def read_congestion_control() -> tuple[str, str, str]:
        """Read the TCP congestion-control settings of this network namespace.

        Reads the sysctl files directly (unprivileged, read-only), so it works inside an
        unprivileged container. Returns (default, allowed, available); each value is a single
        whitespace-free token (spaces in the allowed/available lists become commas).
        Unreadable values come back as "unknown".
        """

        def read(name: str) -> str:
            try:
                with open(f"/proc/sys/net/ipv4/{name}") as f:
                    return f.read().strip().replace(" ", ",") or "unknown"
            except OSError:
                return "unknown"

        return (
            read("tcp_congestion_control"),
            read("tcp_allowed_congestion_control"),
            read("tcp_available_congestion_control"),
        )

    @staticmethod
    def read_tcp_seg_counters() -> tuple[int, int] | None:
        """Return cumulative (OutSegs, RetransSegs) from this namespace's /proc/net/snmp.

        Reads the calling process's own /proc/net/snmp, which is per network namespace: inside a
        container the counts reflect that container's TCP stack rather than the host's. Unprivileged
        and always readable. Returns None if the counters cannot be read or parsed.
        """
        try:
            with open("/proc/net/snmp") as f:
                header = values = None
                for line in f:
                    if line.startswith("Tcp:"):
                        if header is None:
                            header = line.split()[1:]
                        else:
                            values = line.split()[1:]
                            break
            if not header or not values:
                return None
            stats = dict(zip(header, values))
            return int(stats["OutSegs"]), int(stats["RetransSegs"])
        except (OSError, KeyError, ValueError, IndexError):
            return None

    def _log_per_peer_tcp(self) -> None:
        """Log a summary of remote TCP peers plus a detail line for the worst connections.

        Best effort: parses `ss -tin`, skipping loopback peers. The summary line has fixed
        cardinality and is scraped into Prometheus; detail lines are capped at MAX_PEER_TCP_LINES,
        ranked by lifetime retransmits and then rtt inflation over the path minimum. Disables
        itself after the first failure so a missing `ss` binary does not warn every interval.
        """
        if self._per_peer_tcp_disabled:
            return
        try:
            result = subprocess.run(["ss", "-tin"], capture_output=True, text=True, timeout=10)
            lines = result.stdout.splitlines()
        except Exception as e:
            logger.warning(f"Per-peer TCP monitor is not working and will be silenced: {e}")
            self._per_peer_tcp_disabled = True
            return

        peers = self._parse_peer_tcp_stats(lines)
        if not peers:
            return

        def quantile_str(ordered: list[float], frac: float) -> str:
            return f"{self._quantile(ordered, frac):.2f}" if ordered else "-"

        rtts = sorted(p["rtt_ms"] for p in peers if p["rtt_ms"] is not None)
        rates = sorted(p["rate_mbps"] for p in peers if p["rate_mbps"] is not None)
        retrans_total = sum(p["retrans_total"] for p in peers)
        logger.info(
            f"TCP peers: n={len(peers)} rtt_p50_ms={quantile_str(rtts, 0.5)} rtt_p95_ms={quantile_str(rtts, 0.95)} "
            f"retrans_total={retrans_total} delivery_rate_p50_mbps={quantile_str(rates, 0.5)} "
            f"delivery_rate_min_mbps={quantile_str(rates, 0.0)}"
        )

        worst = sorted(peers, key=lambda p: (p["retrans_total"], p["rtt_gap_ms"]), reverse=True)
        for p in worst[:MAX_PEER_TCP_LINES]:
            logger.info(
                f"TCP peer {p['peer']}: cc={p['cc']} rtt={p['rtt']}ms min_rtt={p['min_rtt']}ms "
                f"retrans={p['retrans']} delivery_rate={p['delivery_rate']}"
            )

    @staticmethod
    def _parse_peer_tcp_stats(lines: list[str]) -> list[dict]:
        """Parse `ss -tin` output into one record per non-loopback remote peer."""
        # ss prints each socket as a header row (State ... Peer) followed by an indented detail row
        peers = []
        peer = None
        for line in lines:
            if not line:
                continue
            if not line[0].isspace():
                fields = line.split()
                if fields and fields[0] == "State":  # column-title row
                    peer = None
                    continue
                peer = fields[4] if len(fields) >= 5 else None
                continue
            if peer is None or peer.startswith(LOOPBACK_PREFIXES):
                continue
            detail = line.strip()
            rtt = NetworkMonitor._ss_search(r"\brtt:([\d.]+)", detail)
            min_rtt = NetworkMonitor._ss_search(r"\bminrtt:([\d.]+)", detail)
            retrans = NetworkMonitor._ss_search(r"\bretrans:(\d+/\d+)", detail, "0/0")
            delivery = NetworkMonitor._ss_search(r"\bdelivery_rate (\d+(?:\.\d+)?[KMG]?bps)", detail)
            peers.append(
                {
                    "peer": peer,
                    "cc": detail.split(None, 1)[0] if detail else "unknown",
                    "rtt": rtt,
                    "min_rtt": min_rtt,
                    "retrans": retrans,
                    "delivery_rate": delivery,
                    "rtt_ms": float(rtt) if rtt != "-" else None,
                    "rtt_gap_ms": (float(rtt) - float(min_rtt)) if "-" not in (rtt, min_rtt) else 0.0,
                    "retrans_total": int(retrans.split("/")[1]),
                    "rate_mbps": NetworkMonitor._rate_to_mbps(delivery),
                }
            )
        return peers

    @staticmethod
    def _rate_to_mbps(raw: str) -> float | None:
        match = re.match(r"([\d.]+)([KMG]?)bps", raw)
        if not match:
            return None
        multiplier = {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9}[match.group(2)]
        return float(match.group(1)) * multiplier / 1e6

    @staticmethod
    def _quantile(ordered: list[float], frac: float) -> float:
        idx = min(len(ordered) - 1, round(frac * (len(ordered) - 1)))
        return ordered[idx]

    @staticmethod
    def _ss_search(pattern: str, text: str, default: str = "-") -> str:
        match = re.search(pattern, text)
        return match.group(1) if match else default

    def run(self):
        # Store initial network counters
        logger.info("Running network bandwidth monitor")

        cc_default, cc_allowed, cc_available = self.read_congestion_control()
        logger.info(f"TCP congestion control: default={cc_default} allowed={cc_allowed} available={cc_available}")

        initial_counters = psutil.net_io_counters(pernic=True)
        prev_segs = self.read_tcp_seg_counters()
        start_time = time.time()

        try:
            # Determine interfaces to monitor
            if self.interface:
                interfaces = [self.interface]
            else:
                interfaces = [iface for iface in initial_counters.keys() if iface != "lo"]

            # Monitoring loop
            while True:
                # Wait for the interval
                time.sleep(self.interval)

                # Get current network counters
                current_counters = psutil.net_io_counters(pernic=True)
                current_time = time.time()

                # Calculate time elapsed
                elapsed = current_time - start_time

                # Process each interface
                bytes_sent = 0
                bytes_recv = 0
                for iface in interfaces:
                    if iface not in current_counters:
                        continue

                    # Calculate throughput
                    initial = initial_counters.get(iface, None)
                    if not initial:
                        continue

                    bytes_sent += current_counters[iface].bytes_sent - initial.bytes_sent
                    bytes_recv += current_counters[iface].bytes_recv - initial.bytes_recv

                # Calculate megabytes sent and received per second
                bytes_sent = (bytes_sent) / elapsed / (1024 * 1024)
                bytes_recv = (bytes_recv) / elapsed / (1024 * 1024)

                bits_sent = bytes_sent * 8
                bits_recv = bytes_recv * 8

                # Get current TCP connection counts
                tcp_count = self.get_tcp_connection_count()

                # Log results
                logger.info(
                    f"Time {time.strftime('%Y-%m-%d %H:%M:%S'):<20} "
                    f"Interface Agg "
                    f"Sent {bits_sent:>10.2f} Mbps "
                    f"Rcv {bits_recv:>10.2f} Mbps "
                )

                logger.info(f"Time {time.strftime('%Y-%m-%d %H:%M:%S'):<20} Number open connections {tcp_count}")

                # cc is repeated on this line because the one-time readout above is logged before
                # the Prometheus monitor starts scraping and never reaches it
                current_segs = self.read_tcp_seg_counters()
                if prev_segs is not None and current_segs is not None:
                    out_delta = current_segs[0] - prev_segs[0]
                    retrans_delta = current_segs[1] - prev_segs[1]
                    if out_delta >= 0 and retrans_delta >= 0:  # skip a counter reset / namespace change
                        retrans_pct = (100.0 * retrans_delta / out_delta) if out_delta > 0 else 0.0
                        logger.info(
                            f"TCP retransmit: {retrans_delta} of {out_delta} segs retransmitted "
                            f"({retrans_pct:.3f}%) cc={cc_default}"
                        )
                prev_segs = current_segs

                if self.log_per_peer_tcp:
                    self._log_per_peer_tcp()

                # Update initial counters and start time
                initial_counters = current_counters
                start_time = current_time

                # Check if monitoring duration is specified
                if self.duration and current_time - start_time >= self.duration:
                    break

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
