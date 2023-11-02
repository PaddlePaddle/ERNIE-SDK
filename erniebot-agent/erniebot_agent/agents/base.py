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

import abc
import inspect
import json
from typing import Dict, List, Optional

from erniebot_agent.agents.callback.callback_manager import CallbackManager
from erniebot_agent.agents.callback.default import get_default_callbacks
from erniebot_agent.agents.callback.handlers.base import CallbackHandler
from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.messages import Message
from erniebot_agent.tools.base import Tool
from erniebot_agent.utils.json import to_compact_json, to_pretty_json
from erniebot_agents.memory.base import Memory


class ToolManager(object):
    """A `ToolManager` instance manages tools for an agent.

    This implementation is based on `ToolsManager` in https://github.com/deepset-ai/haystack/blob/main/haystack/agents/base.py
    """

    def __init__(self, tools: List[Tool]) -> None:
        super().__init__()
        self._tools: Dict[str, Tool] = {}
        for tool in tools:
            self.add_tool(tool)

    def add_tool(self, tool: Tool) -> None:
        tool_name = tool.tool_name
        if tool_name in self._tools:
            raise ValueError(f"Tool {repr(tool_name)} is already registered.")
        self._tools[tool_name] = tool

    def remove_tool(self, tool_name: str) -> None:
        if tool_name not in self._tools:
            raise ValueError(f"Tool {repr(tool_name)} is not registered.")
        self._tools.pop(tool_name)

    def get_tool(self, tool_name: str) -> Tool:
        if tool_name not in self._tools:
            raise ValueError(f"Tool {repr(tool_name)} is not registered.")
        return self._tools[tool_name]

    def get_tools(self) -> List[Tool]:
        return list(self._tools.values())

    def get_tool_names(self) -> str:
        return ", ".join(self._tools.keys())

    def get_tool_names_with_descriptions(self) -> str:
        return "\n".join(
            f"{name}:{to_compact_json(tool.function_input())}" for name, tool in self._tools.items()
        )


class Agent(metaclass=abc.ABCMeta):
    llm: ChatModel
    memory: Memory
    _tool_manager: ToolManager
    _callback_manager: CallbackManager


class Agent(metaclass=abc.ABCMeta):
    llm: ChatModel
    memory: Memory
    _tool_manager: ToolManager
    _callback_manager: CallbackManager

    def __init__(
        self,
        llm: ChatModel,
        tools: Union[ToolManager, List[Tool]],
        memory: Memory,
        *,
        callbacks: Optional[List[CallbackHandler]] = None,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.memory = memory
        self._tool_manager = ToolManager(tools)
        self._callback_manager = CallbackManager(callbacks or get_default_callbacks())

    async def run(self, prompt: str) -> str:
        self._callback_manager.on_agent_start(agent=self, prompt=prompt)
        output = await self._run(prompt)
        self._callback_manager.on_agent_end(agent=self, output=output)
        return output

    async def run_tool(self, tool_name: str, tool_args: str) -> str:
        tool = self._tool_manager.get_tool(tool_name)
        self._callback_manager.on_tool_start(agent=self, tool=tool, input_args=tool_args)
        tool_resp = await self._run_tool(tool, tool_args)
        self._callback_manager.on_tool_end(agent=self, tool=tool, response=tool_resp)
        return tool_resp

    async def run_llm(self, messages: List[Message]) -> Message:
        self._callback_manager.on_llm_start(agent=self, llm=self.llm, messages=messages)
        llm_resp = await self._run_llm(messages)
        self._callback_manager.on_llm_end(agent=self, llm=self.llm, response=llm_resp)
        return llm_resp

    def import_tool(self, tool: Tool) -> None:
        self._tool_manager.add_tool(tool)

    def reset(self):
        self.memory.forget()

    @abc.abstractmethod
    def _run(self, prompt: str) -> str:
        raise NotImplementedError

    async def _run_tool(self, tool: Tool, tool_args: str) -> str:
        tool_args = json.loads(tool_args)
        tool_args = self._validate_tool_args(tool, tool_args)
        await tool.run(tool_args)
        tool_resp = to_pretty_json(tool_resp)
        return tool_resp

    async def _run_llm(self, messages: List[Message]) -> Message:
        llm_resp = await self.llm.run(messages, stream=False)
        return llm_resp

    def _validate_tool_args(self, tool: Tool, tool_args: str) -> dict:
        inspect.signature(tool)
