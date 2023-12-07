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

__all__ = ["color_text", "color_msg"]

COLORS = {
    "Purple": "\033[95m",
    "Green": "\033[92m",
    "Yellow": "\033[93m",
    "Red": "\033[91m",
    "RESET": "\033[0m",
    "Blue": "\033[94m",
    None: "",
}


def color_text(text: str, color: Optional[str]) -> str:
    if color is not None and color not in COLORS:
        color_keys = list(COLORS.keys())
        raise ValueError("Only support colors: " + ", ".join(str(key) for key in color_keys))

    if not color:
        return text
    else:
        return COLORS[color] + str(text) + COLORS["RESET"]


def color_msg(message: Union[Message, List[Message]], role_color: dict, max_length: int) -> str:
    res = ""
    if isinstance(message, list):
        for msg in message:
            res += _color_by_role(msg, role_color, max_length)
            res += "\n"
    else:
        res = _color_by_role(message, role_color, max_length)
    return res


def _color_by_role(msg: Message, role_color: dict, max_length: int):
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
                res += f" {k}: {COLORS[possible_color]}{v}{COLORS['RESET']} \n"
            else:
                res += f" {k}: {v} \n"

    return res.strip("\n")
