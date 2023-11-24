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
# limitations under the License

from dataclasses import dataclass
from typing import Dict, Optional, TypedDict


class Message:
    """The base class of a message."""

    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
        self._param_names = ["role", "content"]

    def to_dict(self) -> Dict[str, str]:
        res = {}
        for name in self._param_names:
            res[name] = getattr(self, name)
        return res

    def __str__(self) -> str:
        res = ""
        for name in self._param_names:
            value = getattr(self, name)

            if isinstance(value, dict):
                res += f"{name}: \n"
                for k, v in value.items():
                    res += f"    {k}: {v}, \n"
            elif value is not None and value != "":
                res += f"{name}: \n    {value}, \n"

        return res[:-2]


class SystemMessage(Message):
    """A message from human to set system information."""

    def __init__(self, content: str):
        super().__init__(role="system", content=content)


class HumanMessage(Message):
    """A message from human."""

    def __init__(self, content: str):
        super().__init__(role="user", content=content)


class FunctionCall(TypedDict):
    name: str
    thoughts: str
    arguments: str


class AIMessage(Message):
    """A message from the assistant."""

    def __init__(self, content: str, function_call: Optional[FunctionCall] = None):
        super().__init__(role="assistant", content=content)
        self.function_call = function_call
        self._param_names = ["role", "content", "function_call"]


class FunctionMessage(Message):
    """A message from human that contains the result of a function call."""

    def __init__(self, name: str, content: str):
        super().__init__(role="function", content=content)
        self.name = name
        self._param_names = ["role", "name", "content"]


@dataclass
class AIMessageChunk(object):
    content: str
    function_call: Optional[FunctionCall]
