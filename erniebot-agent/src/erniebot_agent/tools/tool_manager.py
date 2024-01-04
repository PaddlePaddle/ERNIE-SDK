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

import functools
import json
import types
from dataclasses import asdict
from typing import Dict, List, final

import uvicorn
from fastapi import FastAPI

from erniebot_agent.tools.base import BaseTool, Tool
from erniebot_agent.tools.remote_tool import RemoteTool


@final
class ToolManager(object):
    """A `ToolManager` instance manages tools for an agent.

    This implementation is based on `ToolsManager` in
    https://github.com/deepset-ai/haystack/blob/main/haystack/agents/base.py
    """

    def __init__(self, tools: List[BaseTool]) -> None:
        super().__init__()
        self._tools: Dict[str, BaseTool] = {}
        for tool in tools:
            self.add_tool(tool)

    def __getitem__(self, tool_name: str) -> BaseTool:
        return self.get_tool(tool_name)

    def add_tool(self, tool: BaseTool) -> None:
        tool_name = tool.tool_name
        if tool_name in self._tools:
            raise ValueError(f"Name {repr(tool_name)} is already registered.")
        self._tools[tool_name] = tool

    def remove_tool(self, tool: BaseTool) -> None:
        tool_name = tool.tool_name
        if tool_name not in self._tools:
            raise ValueError(f"Name {repr(tool_name)} is not registered.")
        if self._tools[tool_name] is not tool:
            raise RuntimeError(f"The tool with the registered name {repr(tool_name)} is not the given tool.")
        self._tools.pop(tool_name)

    def get_tool(self, tool_name: str) -> BaseTool:
        if tool_name not in self._tools:
            raise ValueError(f"Name {repr(tool_name)} is not registered.")
        return self._tools[tool_name]

    def get_tools(self) -> List[BaseTool]:
        return list(self._tools.values())

    def get_tool_names(self) -> str:
        return ", ".join(self._tools.keys())

    def get_tool_names_with_descriptions(self) -> str:
        return "\n".join(
            f"{name}:{json.dumps(tool.function_call_schema())}" for name, tool in self._tools.items()
        )

    def get_tool_schemas(self):
        return [tool.function_call_schema() for tool in self._tools.values()]

    def serve(self, port: int = 5000):
        """start the local server for toolkit

        Args:
            port (int, optional): the port of local toolkit server. Defaults to 5000.
        """

        app = FastAPI(title="erniebot-agent-tools", version="0.0")

        def create_func(f, func_types, tool):
            # add your code to first parameter
            new_func = types.FunctionType(
                f.__code__, f.__globals__, f.__name__, f.__defaults__, f.__closure__
            )
            new_func.__annotations__ = func_types
            return functools.partial(new_func, __tool__=tool)

        for tool in self._tools.values():
            if not isinstance(tool, Tool):
                continue

            async def create_tool_endpoint_without_inputs(__tool__):
                return await __tool__()

            async def create_tool_endpoint(__tool__, inputs):
                data = asdict(inputs)
                return await __tool__(**data)

            if tool.input_type is not None:
                type_annotation = {"inputs": tool.input_type}
                func = create_func(create_tool_endpoint, type_annotation, tool)
            else:
                func = create_func(create_tool_endpoint_without_inputs, {}, tool)

            tool_name = tool.tool_name.split("/")[-1]
            app.add_api_route(
                f"/erniebot-agent-tools/0.0/{tool_name}",
                endpoint=func,
                response_model=tool.ouptut_type,
                description=tool.description,
                operation_id=tool.tool_name,
            )

        @app.get("/.well-known/openapi.yaml")
        def get_openapi_yaml():
            """get openapi json schema from fastapi"""
            return app.openapi()

        uvicorn.run(app, host="0.0.0.0", port=port)
