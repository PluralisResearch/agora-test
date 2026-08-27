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

from datetime import datetime, timezone

from agora_server.hivemind.utils.logging import TextStyle, always_log_caller


class CustomFormatter(logging.Formatter):
    """A formatter that allows a log time and caller info to be overridden via
    `logger.log(level, message, extra={"origin_created": ..., "caller": ...})`.
    """

    _LEVEL_TO_COLOR = {
        logging.DEBUG: TextStyle.PURPLE,
        logging.INFO: TextStyle.BLUE,
        logging.WARNING: TextStyle.ORANGE,
        logging.ERROR: TextStyle.RED,
        logging.CRITICAL: TextStyle.RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        if hasattr(record, "origin_created"):
            record.created = record.origin_created
            record.msecs = (record.created - int(record.created)) * 1000

        if record.levelno > logging.INFO or always_log_caller:
            if not hasattr(record, "caller"):
                record.caller = f"{record.name}.{record.funcName}:{record.lineno}"
            record.caller_block = f" [{TextStyle.BOLD}{record.caller}{TextStyle.RESET}]"
        else:
            record.caller_block = ""

        # Aliases for the format argument
        record.levelcolor = (
            self._LEVEL_TO_COLOR[record.levelno] if record.levelno in self._LEVEL_TO_COLOR else TextStyle.BLUE
        )
        record.bold = TextStyle.BOLD
        record.reset = TextStyle.RESET

        return super().format(record)


class CustomFileFormatter(logging.Formatter):
    """A formatter that logs time in UTC."""

    converter = lambda *args: datetime.now(timezone.utc).timetuple()

    def format(self, record: logging.LogRecord) -> str:
        if hasattr(record, "origin_created"):
            record.created = record.origin_created
            record.msecs = (record.created - int(record.created)) * 1000

        if record.levelno > logging.INFO:
            if not hasattr(record, "caller"):
                record.caller = f"{record.name}.{record.funcName}:{record.lineno}"
            record.caller_block = f" [{record.caller}]"
        else:
            record.caller_block = ""

        return super().format(record)
