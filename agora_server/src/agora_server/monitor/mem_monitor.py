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
import time

import numpy as np
import psutil
import torch

from agora_server.hivemind.utils.logging import get_logger


logger = get_logger(__name__)


class MemoryTracker(threading.Thread):
    """Monitor memory."""

    def __init__(self, interval: int = 15):
        super().__init__(daemon=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.log_gpumem = True
        if self.device == "cpu":
            logger.info("GPU memory monitor skipped. Device is cpu.")
            self.log_gpumem = False

        self.interval = interval
        if self.log_gpumem:
            self.total_gpu_vram = torch.cuda.get_device_properties(self.device).total_memory / 1024**3  # Gb

        self.total_ram = psutil.virtual_memory().total / 1024**3  # Gb

        self.start()

    def get_gpu_memory_usage(self, device: torch.types.Device | int = 0):
        """Return the current GPU memory usage."""
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # Gb
        reserved = torch.cuda.memory_reserved(device) / 1024**3  # Gb

        return {
            "allocated": allocated,
            "reserved": reserved,
        }

    def get_memory_usage(self):
        """Returns the current RAM usage."""
        ram = psutil.virtual_memory()
        used_ram_gb = ram.used / 1024**3  # Gb

        return {
            "used": used_ram_gb,
        }

    def run(self):
        # Store initial network counters
        logger.info("Running memory monitor")
        start_time = time.time()
        alloc = []
        reserv = []
        ram_used = []
        cpu_used = []

        # Prime cpu_percent so the first real sample reflects activity since the
        # previous call rather than returning 0.0.
        psutil.cpu_percent(interval=None)

        try:
            while True:
                if time.time() - start_time > self.interval:
                    if self.log_gpumem:
                        logger.info(f"GPU mem size is {round(self.total_gpu_vram)}Gb")
                        logger.info(f"Allocated GPU mem is {np.mean(alloc) / self.total_gpu_vram:.2f}")
                        logger.info(f"Reserved GPU mem is {np.mean(reserv) / self.total_gpu_vram:.2f}")
                    logger.info(f"Total RAM mem is {round(self.total_ram)}Gb")
                    logger.info(f"Used RAM mem is {np.mean(ram_used) / self.total_ram:.2f}")
                    logger.info(f"CPU load is {np.mean(cpu_used) / 100:.2f}")
                    start_time = time.time()
                    alloc = []
                    reserv = []
                    ram_used = []
                    cpu_used = []

                if self.log_gpumem:
                    vram_usage = self.get_gpu_memory_usage()
                    alloc.append(vram_usage["allocated"])
                    reserv.append(vram_usage["reserved"])
                ram_usage = self.get_memory_usage()
                ram_used.append(ram_usage["used"])
                cpu_used.append(psutil.cpu_percent(interval=None))
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n GPU monitoring stopped by user.")
