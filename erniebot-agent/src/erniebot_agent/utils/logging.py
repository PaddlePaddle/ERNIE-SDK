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
import re
from typing import Dict, List, Optional, Union

from erniebot_agent.memory import FunctionMessage, Message
from erniebot_agent.utils import config_from_environ as C
from erniebot_agent.utils.json import to_pretty_json
from erniebot_agent.utils.output_style import ColoredContent

__all__ = ["logger", "set_role_color", "setup_logging"]

logger = logging.getLogger("erniebot_agent")


def _handle_color_pattern(s: str):
    """Set ASCII color code into right sequence to avoid color conflict."""
    color_pattern = r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])"
    color_lis = re.findall(color_pattern, s)
    origin_text = re.split(color_pattern, s)

    # Preprocess: Split the text by ASCII color code
    idx_color, idx_text = 0, 0
    while idx_text < len(origin_text):
        if idx_text > 0 and origin_text[idx_text - 1] != "" and origin_text[idx_text] != "":
            origin_text.insert(idx_text, "")
        idx_text += 1

    for i in range(len(origin_text)):
        if origin_text[i] == "":
            origin_text[i] = color_lis[idx_color]
            idx_color += 1

    # Process the wrong sequence
    # Set the color after reset code to previous color
    stack: List[str] = []
    for i in range(len(origin_text)):
        if origin_text[i] in color_lis:
            color = origin_text[i]
            if color == "\033[0m":
                stack.pop()
                if stack:
                    origin_text[i] = stack[-1]
            else:
                stack.append(color)
    return "".join(origin_text)


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
        # color the message
        origin_args = record.args
        if record.args:
            record_lis = list(record.args)
            for i in range(len(record_lis)):
                if isinstance(record_lis[i], ColoredContent):
                    record_lis[i] = record_lis[i].get_colored_text()
            record.args = tuple(record_lis)

        log_message = super(ColorFormatter, self).format(record)

        log_message = _handle_color_pattern(
            self.COLORS.get(record.levelname, "") + log_message + self.COLORS["RESET"]
        )
        record.args = origin_args
        return log_message


class FileFormatter(logging.Formatter):
    def format(self, record):
        output = []
        if record.args:
            for arg in record.args:
                if isinstance(arg, ColoredContent):
                    output = self.extract_content(arg.text, output)
        log_message = ""
        if output:
            log_message += to_pretty_json(output)
        return log_message

    def extract_content(self, text: Union[List[Message], Message, str], output: list) -> List[dict]:
        """Extract the content from message and convert to json format."""
        if isinstance(text, list):
            # List of messages
            chat_lis = []
            func_lis = []
            for i in range(len(text)):
                if isinstance(text[i], Message):
                    chat_res, func_res = self.handle_message(text[i])
                    chat_lis.append(chat_res)
                    if func_res:
                        func_lis.append(func_res)
            output += [{"conversation": chat_lis.copy()}]
            if func_lis:
                output += [{"function_call": func_lis.copy()}]
            return output

        elif isinstance(text, str):
            # Only handle Message Type
            return []
        else:
            # Message type
            chat_res, func_res = self.handle_message(text)
            output += [{"conversation": [chat_res]}]
            if func_res:
                output += [{"function_call": [func_res]}]
            return output

    def handle_message(self, message):
        if isinstance(message, FunctionMessage):
            func_dict = {
                "name": message.name,
                "arguments": message.content,
            }
            return message.to_dict(), func_dict
        else:
            return message.to_dict(), None


def set_role_color(open: bool = True, role_color: Optional[Dict] = None):
    """
    Open or close color role in log, if open, different role will have different color.

    Args:
        open (bool, optional): whether or not to open. Defaults to True.
    """
    if open:
        if not role_color:
            role_color = {"user": "Blue", "function": "Purple", "assistant": "Yellow"}
    else:
        role_color = {}

    ColoredContent.set_global_role_color(role_color)


def setup_logging(
    verbosity: Optional[str] = None,
    use_standard_format: bool = True,
    use_file_handler: bool = False,
    max_log_length: int = 100,
) -> None:
    """Configures logging for the ERNIE Bot Agent library.

    Args:
        verbosity: A value indicating the logging level.
        use_standard_format: If True, use the library's standard logging format.
        use_file_handler: If True, use the library's file handler.
        max_log_length: The max length of log message each round.

    Raises:
        ValueError: If the provided verbosity is not a valid logging level.
    """
    if verbosity is None:
        verbosity = C.get_logging_level()

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
            set_role_color()

        log_file_path = C.get_logging_file_path()
        if log_file_path is None:
            log_file_path = "erniebot-agent.log"
        if use_file_handler or log_file_path:
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(FileFormatter("%(message)s"))
            logger.addHandler(file_handler)

        ColoredContent.set_global_max_length(max_log_length)
