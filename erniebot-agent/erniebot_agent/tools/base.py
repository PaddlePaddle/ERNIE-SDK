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
import ast
from datetime import datetime
from loguru import logger
from typing import Optional, Any
from erniebot_agent.tools.schema import (ToolView, ParametersView,
                                         ParameterView, RemoteToolView,
                                         PluginSchema, scrub_dict)

import docstring_parser
from docstring_parser import Docstring, DocstringParam, DocstringReturns


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
            example.strip()
            for example in docstring.examples[0].description.split(">>>")
            if example.strip()
        ]
        examples_raws = [json.loads(example) for example in examples_raws]

        return ToolView(
            name=self.tool_name,
            description=docstring.short_description or
            docstring.long_description or "",
            parameters=ParametersView.from_docstring(docstring.params),
            returns=ParametersView.from_docstring(docstring.returns),
            examples=examples_raws)

    def function_input(self):
        view = self.view()
        inputs = {
            "name": view.name,
            "description": view.description,
            "parameters": view.parameters.to_function_inputs(),
            "responses": view.returns.to_function_inputs(),
            "examples": view.examples
        }
        return scrub_dict(inputs)


class CalculatorTool(Tool):
    """"""

    def __call__(self, math_formula: str) -> float:
        """你非常擅长将口语化的数学公式描述语言转化标准可执行的数学公式计算

        Args:
            math_formula (str): 标准的数学公式，例如："2+3"、"3 - 4 * 6", "(3 + 4) * (6 + 4)" 等。

        Examples:
            >>> {"role": "user", "content": "请告诉我三加六等于多少"}
            >>> {"role": "assistant", "content": null, "function_call": {"name": "CalculatorTool", "arguments": "{'math_formula':'3+6'}"}}
            >>> {"role": "function", "name": "CalculatorTool", "content": "{'result':9}"}
            >>> {"role": "user", "content": "五乘以一加八等于多少"}
            >>> {"role": "assistant", "content": null, "function_call": {"name": "CalculatorTool", "arguments": "{'math_formula':'5*(1+8)'}"}}
            >>> {"role": "function", "name": "CalculatorTool", "content": "{'result':45}"}
            >>> {"role": "user", "content": "我想知道十二除以 4 再加 5 等于多少"}
            >>> {"role": "assistant", "content": null, "function_call": {"name": "CalculatorTool", "arguments": "{'math_formula':'12/4+5'}"}}
            >>> {"role": "function", "name": "CalculatorTool", "content": "{'result':8}"}

        Returns:
            int: 数学公式计算的结果
        """

        return eval(math_formula)


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
