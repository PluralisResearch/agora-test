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

import logging
import re
import socket
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import median

import psutil
import speedtest
import torch

from pydantic import BaseModel
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from agora_server.hivemind.utils.logging import get_logger
from agora_server.utils.cloud_detect import detect_cloud_provider


logger = get_logger(__name__)


SPEEDTEST_SERVERS = [
    {
        "url": "http://fast.actualbroadband.com:8080/speedtest/upload.php",
        "lat": "45.6018",
        "lon": "-121.1848",
        "name": "The Dalles, OR",
        "country": "United States",
        "cc": "US",
        "sponsor": "Actual Broadband",
        "id": "60580",
        "host": "fast.actualbroadband.com:8080",
    },
    {
        "url": "http://speed.rconnects.com:8080/speedtest/upload.php",
        "lat": "45.2082",
        "lon": "-122.0669",
        "name": "Estacada, OR",
        "country": "United States",
        "cc": "US",
        "sponsor": "Cascade Access, Inc (Reliance Connects)",
        "id": "12013",
        "host": "speed.rconnects.com:8080",
    },
    {
        "url": "http://portland.speedtest.centurylink.net:8080/speedtest/upload.php",
        "lat": "45.5236",
        "lon": "-122.6750",
        "name": "Portland, OR",
        "country": "United States",
        "cc": "US",
        "sponsor": "CenturyLink",
        "id": "10162",
        "host": "portland.speedtest.centurylink.net:8080",
    },
    {
        "url": "http://ptdvuspeedtest01p.allstream.com:8080/speedtest/upload.php",
        "lat": "45.5236",
        "lon": "-122.6750",
        "name": "Portland, OR",
        "country": "United States",
        "cc": "US",
        "sponsor": "allstream",
        "id": "56593",
        "host": "ptdvuspeedtest01p.allstream.com:8080",
    },
    {
        "url": "http://speedtest.sherwoodbroadband.com:8080/speedtest/upload.php",
        "lat": "45.3563",
        "lon": "-122.8559",
        "name": "Sherwood, OR",
        "country": "United States",
        "cc": "US",
        "sponsor": "Sherwood Broadband",
        "id": "33992",
        "host": "speedtest.sherwoodbroadband.com:8080",
    },
]


def _tcp_ping(host: str, port: int, count: int = 3, timeout: float = 3) -> float | None:
    """Measure TCP handshake RTT to host:port, averaged over `count` attempts.

    Returns average RTT in milliseconds, or None if all attempts fail.
    """
    rtts: list[float] = []
    for _ in range(count):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            start = time.monotonic()
            sock.connect((host, port))
            rtt = (time.monotonic() - start) * 1000  # ms
            sock.close()
            rtts.append(rtt)
        except Exception:
            logger.debug(f"TCP ping to {host}:{port} failed")
    if not rtts:
        return None
    return sum(rtts) / len(rtts)


def measure_latency(targets: list[tuple[str, int]], count: int = 3, timeout: float = 3) -> float | None:
    """Measure latency by TCP-pinging multiple host:port targets concurrently.

    Returns the median average RTT across reachable targets, or None if all are unreachable.
    """
    if not targets:
        return None

    results: list[float] = []
    with ThreadPoolExecutor(max_workers=len(targets)) as executor:
        futures = {executor.submit(_tcp_ping, host, port, count, timeout): (host, port) for host, port in targets}
        for future in as_completed(futures):
            host, port = futures[future]
            rtt = future.result()
            if rtt is not None:
                logger.debug(f"TCP ping to {host}:{port}: {rtt:.2f} ms")
                results.append(rtt)
            else:
                logger.debug(f"TCP ping to {host}:{port}: unreachable")

    if not results:
        logger.warning("All ping targets are unreachable, latency measurement unavailable")
        return None

    latency = median(results)
    return latency


class NodeInfo(BaseModel):
    device_name: str
    gpu_memory: float  # GB
    ram: float  # GB
    download_speed: float | None
    upload_speed: float | None
    latency: float | None
    cloud_provider: str = "unknown"


def log_speedtest_results(
    download_speed: float | None,
    upload_speed: float | None,
    latency: float | None,
) -> None:
    """Log speedtest results in the canonical format."""
    if download_speed is not None:
        logger.info(f"Download Speed: {download_speed:.2f} Mbps")
    if upload_speed is not None:
        logger.info(f"Upload Speed: {upload_speed:.2f} Mbps")
    if latency is not None:
        logger.info(f"Latency: {latency:.2f} ms")


@retry(
    wait=wait_exponential(multiplier=1),
    stop=stop_after_attempt(5),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.INFO),
)
def test_internet_speed(
    ping_targets: list[tuple[str, int]],
    find_best_speed_server: bool = False,
) -> tuple[float, float, float | None]:
    """Measure download/upload internet speed and latency via TCP ping."""
    logger.info("Testing internet speed...")

    st = speedtest.Speedtest(secure=True)

    if find_best_speed_server:
        try:
            st.get_best_server(SPEEDTEST_SERVERS)
        except Exception:
            pass

    # Perform the download speed test
    download_speed = st.download() / 1000000  # Convert to Mbps

    # Perform the upload speed test
    upload_speed = st.upload() / 1000000  # Convert to Mbps

    # Measure latency via TCP ping
    latency = measure_latency(ping_targets)

    log_speedtest_results(download_speed, upload_speed, latency)

    return (download_speed, upload_speed, latency)


def get_device_info() -> tuple[str, float, float]:
    """Get device name and memory."""
    ram = float(psutil.virtual_memory().total) / 1024**3

    if not torch.cuda.is_available():
        return "cpu", 0, ram

    device_info = torch.cuda.get_device_properties()
    device_name = device_info.name
    gpu_memory = device_info.total_memory / 1024**3  # GB
    return device_name, gpu_memory, ram


def get_node_info(
    initial_peers: list[str],
    do_speed_test: bool = False,
    find_best_speed_server: bool = False,
) -> NodeInfo:
    """Collect information about the node."""
    device_name, gpu_memory, ram = get_device_info()

    try:
        cloud_provider = detect_cloud_provider()
    except Exception:
        logger.error("An error occurred during cloud provider detection, skipping")
        cloud_provider = "unknown"

    if do_speed_test:
        ping_targets: list[tuple[str, int]] = []
        for initial_peer in initial_peers:
            if match := re.search(r"/ip4/([\d.]+)/tcp/(\d+)", initial_peer):
                ping_targets.append((match.group(1), int(match.group(2))))

        try:
            download_speed, upload_speed, latency = test_internet_speed(ping_targets, find_best_speed_server)
        except Exception:
            logger.error("An error occurred during the speed test, skipping")
            download_speed = upload_speed = latency = None
    else:
        download_speed = upload_speed = latency = None

    node_info = NodeInfo(
        device_name=device_name,
        gpu_memory=gpu_memory,
        ram=ram,
        download_speed=download_speed,
        upload_speed=upload_speed,
        latency=latency,
        cloud_provider=cloud_provider,
    )
    return node_info
