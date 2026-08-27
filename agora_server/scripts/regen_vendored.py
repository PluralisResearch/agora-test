#!/usr/bin/env python3
"""Regenerate the build-time artifacts of the agora_server.hivemind subpackage.

Compiles agora_server/hivemind/proto/*.proto -> *_pb2.py and fetches the p2pd binary into
agora_server/hivemind/_bin/. Normally these run automatically via the hatchling build hook
(hatch_build.py); run this manually for editable (`pip install -e`) dev setups or CI debugging:

    python scripts/regen_vendored.py

Set AGORA_BUILD_P2PD=1 to build p2pd from source (requires Go) instead of downloading it.
"""

import os
import sys


# Make the repo-root hatch_build module importable.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hatch_build import build_vendored  # noqa: E402


if __name__ == "__main__":
    build_vendored()
    print("Regenerated agora_server.hivemind artifacts (proto *_pb2.py + p2pd binary).")
