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

import json
from datetime import datetime
from typing import Any, Optional

import docstring_parser
from docstring_parser import Docstring
from erniebot_agent.tools.schema import ParametersView, ToolView, scrub_dict
from loguru import logger


class Tool:
    name: Optional[str] = None

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

    def view(self) -> ToolView:
        """get tool-view object from Tool

        Returns:
            ToolView: the instance of ToolView
        """

        docstring: Docstring = docstring_parser.parse_from_object(self.__call__)
        examples_raws = [
            example.strip() for example in docstring.examples[0].description.split(">>>") if example.strip()
        ]
        examples_raws = [json.loads(example) for example in examples_raws]

        return ToolView(
            name=self.tool_name,
            description=docstring.short_description or docstring.long_description or "",
            parameters=ParametersView.from_docstring(docstring.params, self.tool_name),
            returns=ParametersView.from_docstring(docstring.returns, self.tool_name),
            examples=examples_raws,
        )

    def function_call_schema(self):
        view = self.view()
        inputs = {
            "name": view.name,
            "description": view.description,
            "parameters": view.parameters.function_call_schema(),
            "responses": view.returns.function_call_schema(),
            "examples": view.examples,
        }
        return scrub_dict(inputs)


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
