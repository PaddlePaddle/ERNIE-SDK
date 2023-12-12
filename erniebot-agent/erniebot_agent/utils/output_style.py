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

from typing import List, Optional, Union

from erniebot_agent.messages import Message
from erniebot_agent.utils.json import to_pretty_json

__all__ = ["ColoredText"]

_COLORS = {
    "Purple": "\033[95m",
    "Green": "\033[92m",
    "Yellow": "\033[93m",
    "Red": "\033[91m",
    "RESET": "\033[0m",
    "Blue": "\033[94m",
    None: "",
}


class ColoredText:
    role_color: dict
    max_length: int

    def __init__(
        self,
        text: Union[str, Message, List[Message]],
        role: Optional[str] = None,
        color: Optional[str] = None,
    ):
        self.text = text
        self.role = role
        self.color = color

    def __str__(self):
        return str(self.text)

    @classmethod
    def set_global_role_color(cls, value: dict):
        cls.role_color = value

    @classmethod
    def set_global_max_length(cls, max_length: int):
        cls.max_length = max_length

    def get_colored_text(self):
        if isinstance(self.text, str):
            if self.color:
                # if user chose the color, use the color
                return self._colorize_text(self.text, self.color)
            elif self.role:
                # if user chose the color, use the role color
                return self._colorize_text(self.text, self.role_color[self.role])
            else:
                raise RuntimeError("Please specify the color or role of the text")
        else:
            return self._colorize_msg(self.text, self.role_color)

    def _colorize_text(self, text: str, color: Optional[str]) -> str:
        if color is not None and color not in _COLORS:
            color_keys = list(_COLORS.keys())
            raise ValueError("Only support colors: " + ", ".join(str(key) for key in color_keys))

        if not color:
            return text
        else:
            return _COLORS[color] + str(text) + _COLORS["RESET"]

    def _colorize_msg(self, message: Union[Message, List[Message]], role_color: dict) -> str:
        max_length = self.max_length if self.max_length else 150
        res = ""
        if isinstance(message, list):
            for msg in message:
                res += self._colorized_by_role(msg, role_color, max_length)
                res += "\n"
        else:
            res = self._colorized_by_role(message, role_color, max_length)
        return res

    def _colorized_by_role(self, msg: Message, role_color: dict, max_length: int):
        res = ""
        for k, v in msg.to_dict().items():
            if isinstance(v, dict):
                v = "\n" + to_pretty_json(v)
            elif isinstance(v, str):
                if len(v) >= max_length:
                    v = v[:max_length] + "..."
            if v:
                possible_color = role_color.get(msg.role)
                if possible_color:
                    res += f" {k}: {_COLORS[possible_color]}{v}{_COLORS['RESET']} \n"
                else:
                    res += f" {k}: {v} \n"

        return res.strip("\n")
