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

import subprocess
import threading
import time

import numpy as np
import torch

from agora_server.hivemind.utils.logging import get_logger


logger = get_logger(__name__)


class GpuUtilizationTracker(threading.Thread):
    """Monitor GPU utilization."""

    def __init__(self, interval: int = 60):
        super().__init__(daemon=True)

        self.interval = interval
        self.device_id = torch.cuda.current_device()

        self.start()

    def run(self):
        logger.info("Running GPU utilization monitor")
        start_time = time.time()
        utilizations = []
        # Latches on the first failed reading so we warn the user once instead of
        # flooding the logs with an identical error every polling interval.
        error_logged = False

        try:
            while True:
                if time.time() - start_time > self.interval:
                    if len(utilizations) > 0:
                        logger.info(f"GPU utilization is {np.mean(utilizations):.2f}")
                        utilizations = []
                    start_time = time.time()

                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                    )

                    util_lines = result.stdout.strip().split("\n")
                    util = float(util_lines[self.device_id])
                    utilizations.append(util)
                except Exception as e:
                    if not error_logged:
                        logger.warning(f"GPU utilization monitor is not working and will be silenced: {e}")
                        error_logged = True

                time.sleep(5)

        except KeyboardInterrupt:
            print("\n GPU utilization monitoring stopped by user.")
