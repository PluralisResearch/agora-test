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

import multiprocessing as mp
import queue
import threading

from logging.handlers import QueueHandler, RotatingFileHandler
from pathlib import Path

from agora_server.logging.formatters import CustomFileFormatter


class FileLogger(threading.Thread):
    """Multiprocess-safe file logger running in a separate thread. Uses RotatingFileHandler."""

    def __init__(self, log_file: str, max_file_size: int = 10 * 1024 * 1024, backup_count: int = 20):
        """Initialize FileLogger.

        Args:
            log_file (str): Path to log file.
            max_file_size (int, optional): Maximum size of log file in bytes. Defaults to 10 * 1024 * 1024.
            backup_count (int, optional): Number of backup log files to keep. Defaults to 20.
        """
        super().__init__(daemon=False)

        # Queue handler to receive log records from other processes
        self._log_queue = mp.Queue()
        self.queue_handler = QueueHandler(self._log_queue)
        self._stop_event = threading.Event()

        # File handler to write log records to file
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        formatter = CustomFileFormatter(
            fmt="{asctime}.{msecs:03.0f} [{levelname}]{caller_block} {message}",
            style="{",
            datefmt="%b %d %H:%M:%S",
        )
        self.file_handler = RotatingFileHandler(log_file, maxBytes=max_file_size, backupCount=backup_count)
        self.file_handler.setFormatter(formatter)

    def run(self):
        while not self._stop_event.is_set():
            try:
                record = self._log_queue.get(timeout=1)
                self.file_handler.handle(record)
                self.file_handler.flush()
            except queue.Empty:
                continue

        # Flush remaining logs after stop signal
        while not self._log_queue.empty():
            try:
                record = self._log_queue.get_nowait()
                self.file_handler.handle(record)
            except queue.Empty:
                break

        self.file_handler.flush()
        self.file_handler.close()

    def stop(self):
        """Signal the logger to stop and flush remaining logs."""
        if self.is_alive():
            self._stop_event.set()
            self.join(timeout=5)  # Wait up to 5 seconds for graceful shutdown
