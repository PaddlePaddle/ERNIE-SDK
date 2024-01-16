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

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from erniebot_agent.memory.messages import Message
from erniebot_agent.tools.schema import ToolParameterView, scrub_dict


class BaseTool(ABC):
    @property
    @abstractmethod
    def tool_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def examples(self) -> List[Message]:
        raise NotImplementedError

    @abstractmethod
    async def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def function_call_schema(self) -> dict:
        raise NotImplementedError


class Tool(BaseTool, ABC):
    description: str
    name: Optional[str] = None
    input_type: Optional[Type[ToolParameterView]] = None
    ouptut_type: Optional[Type[ToolParameterView]] = None

    def __str__(self) -> str:
        name = self.name if self.name else self.tool_name
        return "<name: {0}, description: {1}>".format(name, self.description)

    def __repr__(self):
        return self.__str__()

    @property
    def tool_name(self):
        return self.name or self.__class__.__name__

    @abstractmethod
    async def __call__(self, *args: Any, **kwds: Any) -> Dict[str, Any]:
        """the body of tools

        Returns:
            Any:
        """
        raise NotImplementedError

    def function_call_schema(self) -> dict:
        inputs = {
            "name": self.tool_name,
            "description": self.description,
        }

        if len(self.examples) > 0:
            inputs["examples"] = [example.to_dict() for example in self.examples]

        if self.input_type is not None:
            inputs["parameters"] = self.input_type.function_call_schema()
        else:
            inputs["parameters"] = {"type": "object", "properties": {}}

        if self.ouptut_type is not None:
            inputs["responses"] = self.ouptut_type.function_call_schema()

        return scrub_dict(inputs) or {}

    @property
    def examples(self) -> List[Message]:
        return []
