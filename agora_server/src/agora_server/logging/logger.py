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

from logging.handlers import RotatingFileHandler
from pathlib import Path

from agora_server.hivemind.utils import get_logger, use_hivemind_log_handler
from agora_server.logging.file_logger import FileLogger
from agora_server.logging.formatters import CustomFileFormatter, CustomFormatter


class PluralisLogger:
    def __init__(
        self,
        log_level: str = "INFO",
        log_file: str | None = None,
        multiprocess_logging: bool = False,
        file_log_args: dict | None = None,
    ):
        """Initialize logger.

        Args:
            log_level (str, optional): Logging level. Defaults to "INFO".
            log_file (str | None, optional): File to save logs. Defaults to None.
            multiprocess_logging (bool, optional): If True, create FileLogger that is multiprocess-safe. Defaults to False.
            file_log_args (dict | None, optional): Additional args for file logger. Defaults to None.
        """
        # Add extra log level
        EXTRA_LEVEL = 15  # between INFO (20) and DEBUG (10)
        logging.addLevelName(EXTRA_LEVEL, "EXTRA")

        # Create a custom method for the logger
        def extra(self, message, *args, **kwargs):
            if self.isEnabledFor(EXTRA_LEVEL):
                self._log(EXTRA_LEVEL, message, args, **kwargs)

        logging.Logger.extra = extra
        use_hivemind_log_handler("in_root_logger")

        # Convert log level string to logging constant
        numeric_level = getattr(logging, log_level.upper(), log_level)

        # Configure root logger
        self.root_logger = get_logger()
        self.root_logger.setLevel(numeric_level)

        # Set custom loglevel for TaskPool and Runtime to capture worker queue stats
        # taskpool_logger = get_logger("hivemind.moe.server.task_pool")
        # taskpool_logger.setLevel(logging.DEBUG)
        # runtime_logger = get_logger("agora_server.core.server.runtime")
        # runtime_logger.setLevel(logging.DEBUG)

        formatter = CustomFormatter(
            fmt="{asctime}.{msecs:03.0f} [{bold}{levelcolor}{levelname}{reset}]{caller_block} {message}",
            style="{",
            datefmt="%b %d %H:%M:%S",
        )

        for handler in list(self.root_logger.handlers):
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, RotatingFileHandler):
                handler.setFormatter(formatter)

        self.file_logger = None
        if log_file:
            if multiprocess_logging:
                self.start_file_logger(
                    log_file,
                    **(file_log_args if file_log_args is not None else {}),
                )
            else:
                Path(log_file).parent.mkdir(parents=True, exist_ok=True)
                file_formatter = CustomFileFormatter(
                    fmt="{asctime}.{msecs:03.0f} [{levelname}]{caller_block} {message}",
                    style="{",
                    datefmt="%b %d %H:%M:%S",
                )
                max_file_size = (
                    file_log_args.get("max_file_size", 10 * 1024 * 1024) if file_log_args else 10 * 1024 * 1024
                )
                backup_count = file_log_args.get("backup_count", 20) if file_log_args else 20
                file_handler = RotatingFileHandler(log_file, maxBytes=max_file_size, backupCount=backup_count)
                file_handler.setFormatter(file_formatter)
                self.add_handler(file_handler)

    def add_handler(self, handler: logging.Handler):
        """Attach handler to root logger."""
        self.root_logger.addHandler(handler)

    def start_file_logger(self, log_file: str, max_file_size: int = 10 * 1024 * 1024, backup_count: int = 20):
        """Start file logger in a separate thread.

        Args:
            log_file (str): Path to log file.
            max_file_size (int, optional): Maximum size of log file in bytes. Defaults to 10 * 1024 * 1024.
            backup_count (int, optional): Number of backup log files to keep. Defaults to 20.
        """
        self.file_logger = FileLogger(log_file, max_file_size, backup_count)
        self.add_handler(self.file_logger.queue_handler)
        self.file_logger.start()

    def stop(self):
        """Stop the file logger and flush all remaining logs."""
        if self.file_logger is not None:
            self.file_logger.stop()
