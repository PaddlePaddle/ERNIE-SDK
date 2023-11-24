# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import os
from typing import Optional

__all__ = ["logger", "setup_logging"]

logger = logging.getLogger("erniebot_agent")


class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[95m",  # Purple
        "INFO": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[91m" + "\033[1m",  # Bold Red
        "RESET": "\033[0m",
    }

    def format(self, record):
        log_message = super(ColorFormatter, self).format(record)
        log_message = self.COLORS.get(record.levelname, "") + log_message + self.COLORS["RESET"]
        return log_message


def setup_logging(verbosity: Optional[str] = None, use_standard_format: bool = True) -> None:
    """Configures logging for the ERNIE Bot Agent library.

    Args:
        verbosity: A value indicating the logging level.
        use_standard_format: If True, use the library's standard logging format.

    Raises:
        ValueError: If the provided verbosity is not a valid logging level.
    """
    global logger

    if verbosity is None:
        verbosity = os.environ.get("EB_LOGGING_LEVEL", None)

    if verbosity is not None:
        numeric_level = getattr(logging, verbosity.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid logging level: {verbosity}")

        logger.setLevel(numeric_level)
        logger.propagate = False
        if use_standard_format and not logger.hasHandlers():
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(ColorFormatter("%(levelname)s - %(message)s"))
            logger.addHandler(console_handler)
