# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
import sys
from typing import Any, Optional

import colorlog

from erniebot.constants import LOGGER_NAME

__all__ = ["debug", "info", "warning", "error", "critical", "setup_logging"]

_LOG_CONFIG = {
    "DEBUG": {"color": "purple"},
    "INFO": {"color": "green"},
    "WARNING": {"color": "yellow"},
    "ERROR": {"color": "red"},
    "CRITICAL": {"color": "bold_red"},
}

_logger: logging.Logger = logging.getLogger(LOGGER_NAME)


def debug(msg: object, *args: object, **kwargs: Any) -> None:
    _logger.debug(msg, *args, **kwargs)


def info(msg: object, *args: object, **kwargs: Any) -> None:
    _logger.info(msg, *args, **kwargs)


def warning(msg: object, *args: object, **kwargs: Any) -> None:
    _logger.warning(msg, *args, **kwargs)


def error(msg: object, *args: object, **kwargs: Any) -> None:
    _logger.error(msg, *args, **kwargs)


def critical(msg: object, *args: object, **kwargs: Any) -> None:
    _logger.critical(msg, *args, **kwargs)


def setup_logging(verbosity: Optional[str] = None) -> None:
    if verbosity is None:
        verbosity = os.environ.get("EB_LOGGING_LEVEL", None)
    if verbosity is not None:
        _configure_logger(_logger, verbosity.upper())


def _configure_logger(logger: logging.Logger, verbosity: str) -> None:
    if verbosity == "DEBUG":
        _logger.setLevel(logging.DEBUG)
    elif verbosity == "INFO":
        _logger.setLevel(logging.INFO)
    elif verbosity == "WARNING":
        _logger.setLevel(logging.WARNING)
    logger.propagate = False
    if not logger.hasHandlers():
        _add_handler(logger)


def _add_handler(logger: logging.Logger) -> None:
    format = colorlog.ColoredFormatter(
        "%(log_color)s[%(asctime)-15s] [%(levelname)8s]%(reset)s - %(message)s",
        log_colors={key: conf["color"] for key, conf in _LOG_CONFIG.items()},
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(format)
    logger.addHandler(handler)
