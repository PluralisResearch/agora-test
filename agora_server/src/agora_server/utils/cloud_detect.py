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
import shutil
import socket
import subprocess
import urllib.error
import urllib.request

from collections.abc import Callable
from pathlib import Path


GPU_RENTAL_DETECTORS: list[tuple[str, Callable[[], bool]]] = [
    # RunPod - reliable env vars
    (
        "runpod",
        lambda: bool(
            os.environ.get("RUNPOD_POD_ID")
            or os.environ.get("RUNPOD_POD_HOSTNAME")
            or os.environ.get("RUNPOD_API_KEY")
        ),
    ),
    # Vast.ai - container labels and env vars
    (
        "vastai",
        lambda: bool(
            os.environ.get("VAST_CONTAINERLABEL")
            or os.environ.get("VAST_TCP_PORT_22")
            or Path("/.vast_containerlabel").exists()
            or Path("/etc/vastai_kaalia").exists()
        ),
    ),
    # Modal - serverless GPU
    (
        "modal",
        lambda: bool(os.environ.get("MODAL_TASK_ID") or os.environ.get("MODAL_IMAGE_ID") or Path("/modal").exists()),
    ),
    # Replicate - model serving
    ("replicate", lambda: bool(os.environ.get("REPLICATE_PREDICTION_ID") or os.environ.get("COG_PREDICTION_ID"))),
    # Google Colab
    (
        "colab",
        lambda: bool(
            os.environ.get("COLAB_GPU")
            or os.environ.get("COLAB_RELEASE_TAG")
            or Path("/usr/local/lib/python3.10/dist-packages/google/colab").exists()
            or Path("/usr/local/lib/python3.11/dist-packages/google/colab").exists()
        ),
    ),
    # Kaggle Kernels
    ("kaggle", lambda: bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"))),
    # Paperspace / Gradient
    (
        "paperspace",
        lambda: bool(
            os.environ.get("PAPERSPACE_CLUSTER_ID")
            or os.environ.get("PAPERSPACE_METRIC_WORKLOAD_ID")
            or os.environ.get("PS_API_KEY")
            or Path("/etc/paperspace").exists()
            or "paperspacegradient" in socket.getfqdn().lower()
        ),
    ),
    # Lambda Labs - guest agent and hostname patterns
    (
        "lambda_labs",
        lambda: bool(
            Path("/etc/lambda-guest-agent").exists()
            or Path("/usr/lib/lambda-stack").exists()
            or Path("/var/lib/lambda").exists()
        ),
    ),
    # CoreWeave - Kubernetes-native
    (
        "coreweave",
        lambda: bool(
            os.environ.get("COREWEAVE_NODE_NAME")
            or os.environ.get("CW_NODE_NAME")
            or "coreweave" in socket.getfqdn().lower()
        ),
    ),
    # TensorDock
    ("tensordock", lambda: bool(os.environ.get("TENSORDOCK_INSTANCE_ID") or "tensordock" in socket.getfqdn().lower())),
    # Hyperstack
    ("hyperstack", lambda: bool(os.environ.get("HYPERSTACK_INSTANCE_ID") or "hyperstack" in socket.getfqdn().lower())),
    # JarvisLabs
    ("jarvislabs", lambda: bool(os.environ.get("JARVIS_INSTANCE_ID") or "jarvislabs" in socket.getfqdn().lower())),
    # Thunder Compute
    (
        "thunder_compute",
        lambda: bool(os.environ.get("TNR_INSTANCE_ID") or "thundercompute" in socket.getfqdn().lower()),
    ),
    # Nebius AI Cloud
    ("nebius", lambda: bool(os.environ.get("NEBIUS_INSTANCE_ID") or "nebius" in socket.getfqdn().lower())),
    # Cudo Compute
    ("cudo", lambda: bool(os.environ.get("CUDO_VM_ID") or "cudocompute" in socket.getfqdn().lower())),
]


DMI_VENDOR_MAP = {
    "amazon": "aws",
    "google": "gcp",
    "microsoft": "azure",
    "digitalocean": "digitalocean",
    "oracle": "oci",
    "alibaba": "alibaba",
    "hetzner": "hetzner",
    "ovh": "ovh",
    "scaleway": "scaleway",
}


METADATA_ENDPOINTS = [
    ("aws", "http://169.254.169.254/latest/meta-data/", {}),
    ("gcp", "http://metadata.google.internal/computeMetadata/v1/", {"Metadata-Flavor": "Google"}),
    ("azure", "http://169.254.169.254/metadata/instance?api-version=2021-02-01", {"Metadata": "true"}),
]


def _read(path: str) -> str:
    """Read a file, returning '' on any failure. Lowercased and stripped."""
    try:
        return Path(path).read_text().strip().lower()
    except (FileNotFoundError, PermissionError, OSError):
        return ""


def _detect_virt() -> str:
    """Return systemd-detect-virt output, or '' if unavailable.

    Common values: 'none' (bare metal), 'kvm', 'xen', 'vmware', 'hyperv',
    'qemu', 'amazon', 'microsoft', 'docker', 'lxc'.
    """
    if not shutil.which("systemd-detect-virt"):
        return ""
    try:
        result = subprocess.run(
            ["systemd-detect-virt"],
            capture_output=True,
            text=True,
            timeout=1,
        )
        return result.stdout.strip().lower()
    except Exception:
        return ""


def _try_metadata(url: str, headers: dict, timeout: float) -> bool:
    """Return True if the metadata endpoint responds (any HTTP status)."""
    try:
        req = urllib.request.Request(url, headers=headers)
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except urllib.error.HTTPError:
        # An HTTP error still confirms the endpoint exists (e.g. 401 from IMDSv2)
        return True
    except Exception:
        return False


def _detect_serverless() -> str:
    if os.environ.get("LAMBDA_TASK_ROOT") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        return "aws_lambda"
    if os.environ.get("K_SERVICE") and os.environ.get("K_REVISION"):
        return "gcp_cloudrun"
    if os.environ.get("FUNCTIONS_WORKER_RUNTIME"):
        return "azure_functions"
    return ""


def detect_cloud_provider(timeout: float = 0.5) -> str:
    """
    Detect the cloud provider or GPU rental platform.

    Args:
        timeout: Seconds to wait for metadata endpoint responses.

    Returns:
        Provider name (str), 'unknown_cloud', or 'local'.
    """
    # 1. GPU rental platforms (most specific, all checked via env/files)
    for name, check in GPU_RENTAL_DETECTORS:
        try:
            if check():
                return name
        except Exception:
            continue

    # 2. Serverless runtimes
    serverless = _detect_serverless()
    if serverless:
        return serverless

    # 3. Major cloud providers via DMI
    sys_vendor = _read("/sys/class/dmi/id/sys_vendor")
    for key, provider in DMI_VENDOR_MAP.items():
        if key in sys_vendor:
            return provider

    # 4. Cloud metadata endpoints (network calls, last specific check)
    for provider, url, headers in METADATA_ENDPOINTS:
        if _try_metadata(url, headers, timeout):
            return provider

    # 5. Generic: virtualized + hosted-looking, or local?
    virt_type = _detect_virt()

    cloud_ish_signals = [
        virt_type in {"kvm", "xen", "vmware", "hyperv", "qemu", "amazon", "microsoft"},
        Path("/.dockerenv").exists(),
        os.environ.get("KUBERNETES_SERVICE_HOST") is not None,
        "container" in _read("/proc/1/cgroup"),
    ]

    local_signals = [
        virt_type == "none",
        _read("/sys/class/dmi/id/chassis_type") in ("9", "10", "14"),
        any(
            Path(p).exists()
            for p in (
                "/sys/class/power_supply/BAT0",
                "/sys/class/power_supply/BAT1",
            )
        ),
    ]

    if any(local_signals) and not any(cloud_ish_signals):
        return "local"
    if any(cloud_ish_signals):
        return "unknown_cloud"
    return "unknown"
