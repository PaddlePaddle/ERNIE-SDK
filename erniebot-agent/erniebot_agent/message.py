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
from typing import Dict, Optional

from erniebot.response import EBResponse


class Message:
    """The base class of message."""

    def __init__(self, role: str, content: Optional[str]):
        self.role = role
        self.content = content
        self._param_names = ["role", "content"]

    def to_dict(self) -> Dict[str, str]:
        res = {}
        for name in self._param_names:
            value = getattr(self, name)
            if value is not None and value != "":
                res[name] = value
        return res

    def __str__(self) -> str:
        res = ""
        for name in self._param_names:
            value = getattr(self, name)
            if value is not None and value != "":
                res += f"{name}: {value}, "
        return res[:-2]


class SystemMessage(Message):
    """The message from human to set system information."""

    def __init__(self, content: str):
        super().__init__(role="system", content=content)


class HumanMessage(Message):
    """The message from human."""

    def __init__(self, content: str):
        super().__init__(role="user", content=content)


class AIMessage(Message):
    """The message from assistant."""

    def __init__(self, content: Optional[str], function_call: Optional[Dict[str, str]]):
        super().__init__(role="assistant", content=content)
        self.function_call = function_call
        self._param_names = ["role", "content", "function_call"]

    @classmethod
    def from_response(cls, response: EBResponse):
        if hasattr(response, "function_call"):
            return cls(content=None, function_call=response.function_call)
        else:
            return cls(content=response.result, function_call=None)


class FunctionMessage(Message):
    """The message from human to set the result of function call."""

    def __init__(self, name: str, content: str):
        super().__init__(role="function", content=content)
        self.name = name
        self._param_names = ["role", "name", "content"]
