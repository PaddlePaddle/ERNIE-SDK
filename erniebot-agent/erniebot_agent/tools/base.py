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

from datetime import datetime
from typing import Any, List, Optional, Type

from erniebot_agent.message import Message
from erniebot_agent.tools.schema import ToolParameterView, scrub_dict
from loguru import logger


class Tool:
    name: Optional[str] = None
    description: str = "CalculatorTool用于执行数学公式计算"
    inputs: Optional[Type[ToolParameterView]] = None
    ouptuts: Optional[Type[ToolParameterView]] = None

    @property
    def tool_name(self):
        return self.name or self.__class__.__name__

    def __init__(self) -> None:
        self.logger = logger

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """the body of tools

        Returns:
            Any:
        """
        raise NotImplementedError

    def validate_args(self):
        pass

    def function_call_schema(self) -> dict:
        inputs = {
            "name": self.tool_name,
            "description": self.description,
            "examples": [example.to_dict() for example in self.examples],
        }
        if self.inputs is not None:
            inputs["parameters"] = self.inputs.function_call_schema()
        if self.ouptuts is not None:
            inputs["responses"] = self.ouptuts.function_call_schema()

        return scrub_dict(inputs) or {}

    @property
    def examples(self) -> List[Message]:
        return []


class CurrentTimeTool(Tool):
    def __init__(self) -> None:
        super().__init__()
        self.llm = None

    def __call__(self) -> datetime:
        """你是一个功能强大的个人助手，擅长回复当前时间信息，比如：现在是几点钟、现在是几点几分等

        Examples:
            >>> {"role": "user", "content": "现在几点钟？"}
            >>> {"role": "assistant", "content": null, "function_call": {"name": "CurrentTimeTool"}}
            >>> {"role": "function", "name": "CurrentTimeTool", "content": "{'result': '现在是下午五点钟'}"}
            >>> {"role": "user", "content": "麻烦告诉我现在详细的时间信息"}
            >>> {"role": "assistant", "content": null, "function_call": {"name": "CurrentTimeTool"}}
            >>> {"role": "function", "name": "CurrentTimeTool", "content": "{'result':'现在是下午五点三十二分'}"}

        Returns:
            datetime: 当前时间
        """

        return datetime.now()

    def _get_examples(self) -> list[dict]:
        return [
            {"role": "user", "content": "麻烦告诉我现在详细的时间信息"},
            {"role": "assistant", "content": None, "function_call": {"name": "CurrentTimeTool"}},
            {"role": "function", "name": "CurrentTimeTool", "content": "{'result':'现在是下午五点三十二分'}"},
        ]
