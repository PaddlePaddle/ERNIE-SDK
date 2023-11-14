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
from typing import Any


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


class AgentLogger(object):
    def __init__(self, name=None, log_level="INFO") -> None:
        """
        此logger主要用于
        1. 产生error类的日志，包括debug、info、warning、error、critical等。
        2. 产生agent类的日志，用于输出agent的中间输出。

        Args:
            name: 日志记录器的名称，默认为 'Agent Logger'。
        """
        self.logger = logging.getLogger(name if name else "Agent Logger")
        self.set_log_level(log_level)

        # 删除之前的handler以免重复输出
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)

        # 添加默认的handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColorFormatter("%(levelname)s - %(message)s"))

        self.logger.addHandler(console_handler)
        self.logger.addHandler(logging.NullHandler())

    def debug(self, msg: object, *args: object, **kwargs: Any) -> None:
        return self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: object, *args: object, **kwargs: Any) -> None:
        return self.logger.info(msg, *args, **kwargs)

    def agent_info(
        self, msg: object, *args: object, level: str = "LLM", state: str = "START", **kwargs: Any
    ) -> None:
        msg = f"[{level}][{state}]{msg}"
        return self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: object, *args: object, **kwargs: Any) -> None:
        return self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: object, *args: object, **kwargs: Any) -> None:
        return self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: object, *args: object, **kwargs: Any) -> None:
        return self.logger.critical(msg, *args, **kwargs)

    def set_log_level(self, level: str) -> None:
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {level}")
        self.logger.setLevel(numeric_level)

    def add_handler(self, handler) -> None:
        self.logger.addHandler(handler)


if __name__ == "__main__":
    agentlogger = AgentLogger()
    agentlogger.debug("调试日志")
    agentlogger.info("消息日志")
    agentlogger.warning("警告日志")
    agentlogger.error("错误日志 ")
    agentlogger.critical("严重错误日志")
    agentlogger.agent_info(
        "Agent %s starts running with input: %s", "WEATHER", "PROMPT", level="llm", state="start"
    )
    agentlogger.agent_info(
        "Agent %s starts running with input: %s", "WEATHER", "PROMPT", level="llm", state="end"
    )
