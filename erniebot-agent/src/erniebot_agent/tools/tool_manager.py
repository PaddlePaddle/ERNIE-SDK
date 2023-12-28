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

import json
from typing import Dict, Iterable, List, final

from erniebot_agent.tools.base import BaseTool


@final
class ToolManager(object):
    """A `ToolManager` instance manages tools for an agent.

    This implementation is based on `ToolsManager` in
    https://github.com/deepset-ai/haystack/blob/main/haystack/agents/base.py
    """

    def __init__(self, tools: Iterable[BaseTool]) -> None:
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
