"""Hatchling build hook for the agora_server.hivemind subpackage.

Performs the two build-time steps this subpackage requires:

  1. Compile agora_server/hivemind/proto/*.proto -> *_pb2.py (rewriting cross-proto imports to be
     package-relative), and
  2. Download + SHA256-verify the libp2p `p2pd` daemon binary into agora_server/hivemind/_bin/.

Both generated artifacts (the *_pb2.py files and the p2pd binary) are .gitignored and produced at
build time; `[tool.hatch.build.targets.wheel].artifacts` in pyproject.toml ensures they are still
packaged into the wheel.

The same work can be run outside a build (e.g. for editable `pip install -e` dev setups or CI
debugging) via `python scripts/regen_vendored.py`.
"""

from __future__ import annotations

import glob
import hashlib
import os
import platform
import re
import subprocess
import tarfile
import tempfile
import urllib.request


HERE = os.path.abspath(os.path.dirname(__file__))
PKG_ROOT = os.path.join(HERE, "src", "agora_server", "hivemind")
PROTO_DIR = os.path.join(PKG_ROOT, "proto")
BIN_DIR = os.path.join(PKG_ROOT, "_bin")
P2PD_BINARY_PATH = os.path.join(BIN_DIR, "p2pd")

# Pinned to the p2pd release the swarm runs, so peers stay wire/daemon compatible.
P2PD_VERSION = "v0.5.0.hivemind1"
P2PD_SOURCE_URL = f"https://github.com/learning-at-home/go-libp2p-daemon/archive/refs/tags/{P2PD_VERSION}.tar.gz"
P2PD_BINARY_URL = f"https://github.com/learning-at-home/go-libp2p-daemon/releases/download/{P2PD_VERSION}/"
# sha256 of the binary from the release page
P2P_BINARY_HASH = {
    "p2pd-darwin-amd64": "fe00f9d79e8e4e4c007144d19da10b706c84187b3fb84de170f4664c91ecda80",
    "p2pd-darwin-arm64": "0404981a9c2b7cab5425ead2633d006c61c2c7ec85ac564ef69413ed470e65bd",
    "p2pd-linux-amd64": "42f8f48e62583b97cdba3c31439c08029fb2b9fc506b5bdd82c46b7cc1d279d8",
    "p2pd-linux-arm64": "046f18480c785a84bdf139d7486086d379397ca106cb2f0191598da32f81447a",
}


def _sha256(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def proto_compile(output_path: str = PROTO_DIR) -> None:
    """Compile *.proto in `output_path` to *_pb2.py with package-relative imports."""
    import grpc_tools.protoc

    proto_files = glob.glob(os.path.join(output_path, "*.proto"))
    cli_args = [
        "grpc_tools.protoc",
        f"--proto_path={output_path}",
        f"--python_out={output_path}",
    ] + proto_files
    code = grpc_tools.protoc.main(cli_args)
    if code:
        raise ValueError(f"{' '.join(cli_args)} finished with exit code {code}")

    # Make pb2 imports in generated scripts relative (e.g. `import dht_pb2` -> `from . import dht_pb2`)
    for script in glob.iglob(os.path.join(output_path, "*_pb2.py")):
        with open(script, "r+") as file:
            text = file.read()
            file.seek(0)
            file.write(re.sub(r"\n(import .+_pb2.*)", "from . \\1", text))
            file.truncate()


def build_p2p_daemon() -> None:
    """Build p2pd from source (requires Go >= 1.13). Used only when downloading is not possible."""
    from packaging.version import parse as parse_version

    result = subprocess.run("go version", capture_output=True, shell=True).stdout.decode("ascii", "replace")
    m = re.search(r"^go version go([\d.]+)", result)
    if m is None:
        raise FileNotFoundError("Could not find golang installation")
    if parse_version(m.group(1)) < parse_version("1.13"):
        raise OSError(f"Newer version of go required: must be >= 1.13, found {m.group(1)}")

    os.makedirs(BIN_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory() as tempdir:
        dest = os.path.join(tempdir, "libp2p-daemon.tar.gz")
        urllib.request.urlretrieve(P2PD_SOURCE_URL, dest)
        with tarfile.open(dest, "r:gz") as tar:
            tar.extractall(tempdir)
        result = subprocess.run(
            ["go", "build", "-o", P2PD_BINARY_PATH],
            cwd=os.path.join(tempdir, f"go-libp2p-daemon-{P2PD_VERSION.lstrip('v')}", "p2pd"),
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to build p2pd: exited with status code: {result.returncode}")


def download_p2p_daemon() -> None:
    """Download + verify the precompiled p2pd binary for the current platform (no-op if valid)."""
    os.makedirs(BIN_DIR, exist_ok=True)
    arch = platform.machine()
    if arch in ("x86_64", "x64"):
        arch = "amd64"
    if arch in ("aarch64", "aarch64_be", "armv8b", "armv8l"):
        arch = "arm64"
    binary_name = f"p2pd-{platform.system().lower()}-{arch}"

    if binary_name not in P2P_BINARY_HASH:
        raise RuntimeError(
            f"agora_server does not provide a precompiled p2pd binary for {platform.system()} ({arch}). "
            f"Install Go and build it from source (set AGORA_BUILD_P2PD=1)."
        )
    expected_hash = P2P_BINARY_HASH[binary_name]

    if _sha256(P2PD_BINARY_PATH) != expected_hash:
        binary_url = os.path.join(P2PD_BINARY_URL, binary_name)
        print(f"Downloading {binary_url} to {P2PD_BINARY_PATH}")
        urllib.request.urlretrieve(binary_url, P2PD_BINARY_PATH)
        os.chmod(P2PD_BINARY_PATH, 0o777)
        actual_hash = _sha256(P2PD_BINARY_PATH)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"The sha256 checksum for p2pd does not match (expected: {expected_hash}, actual: {actual_hash})"
            )


def build_vendored() -> None:
    """Run both build-time steps. Safe to call repeatedly (idempotent)."""
    proto_compile()
    if os.environ.get("AGORA_BUILD_P2PD") == "1":
        build_p2p_daemon()
    else:
        download_p2p_daemon()


try:
    from hatchling.builders.hooks.plugin.interface import BuildHookInterface

    class CustomBuildHook(BuildHookInterface):
        def initialize(self, version, build_data):  # noqa: ARG002
            build_vendored()

except ImportError:  # hatchling not present (e.g. running the module standalone)
    pass


if __name__ == "__main__":
    build_vendored()
