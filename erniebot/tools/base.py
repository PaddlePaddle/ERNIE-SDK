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

import ast
from loguru import logger
from typing import Optional, Any
from erniebot.tools.schema import ToolView, ParametersView, ParameterView, RemoteToolView, PluginSchema

import docstring_parser
from docstring_parser import Docstring, DocstringParam, DocstringReturns


class Tool:
    name: Optional[str] = None

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
        return ToolView(
            name=self.name or self.__class__.__name__,
            description=docstring.short_description or docstring.long_description or "",
            parameters=ParametersView.from_docstring(docstring.params),
            returns=ParametersView.from_docstring(docstring.returns),
        )


class CalculatorTool(Tool):
    """"""
    name = "calculator"
    async def __call__(self, command: str) -> float:
        """你非常擅长将口语化的数学公式描述语言转化标准可执行的数学公式计算

        Examples:
            >>> ["计算死"]

        Args:
            command (str): 标准的数学公式，例如：2+3、3 - 4 * 6, (3 + 4) * (6 + 4) 等。

        Returns:
            float: 数学公式计算的结果
        """
        
        return ast.literal_eval(command)

