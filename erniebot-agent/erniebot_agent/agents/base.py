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
from typing import Any, Dict, List, Optional, Union, final

from erniebot_agent.agents.callback.callback_manager import CallbackManager
from erniebot_agent.agents.callback.default import get_default_callbacks
from erniebot_agent.agents.callback.handlers.base import CallbackHandler
from erniebot_agent.agents.schema import AgentResponse
from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.memory.base import Memory
from erniebot_agent.messages import AIMessage, Message
from erniebot_agent.tools.base import Tool


@final
class ToolManager(object):
    """A `ToolManager` instance manages tools for an agent.

    This implementation is based on `ToolsManager` in
    https://github.com/deepset-ai/haystack/blob/main/haystack/agents/base.py
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
        return "\n".join(f"{name}:{json.dumps(tool.function_input())}" for name, tool in self._tools.items())

    def get_tool_function_inputs(self):
        return [tool.function_input() for tool in self._tools.values()]


class BaseAgent(metaclass=abc.ABCMeta):
    llm: ChatModel
    memory: Memory
    _tool_manager: ToolManager
    _callback_manager: CallbackManager

    @abc.abstractmethod
    async def async_run(self, prompt: str) -> AgentResponse:
        raise NotImplementedError


class Agent(BaseAgent):
    def __init__(
        self,
        llm: ChatModel,
        tools: Union[ToolManager, List[Tool]],
        memory: Memory,
        *,
        callbacks: Optional[Union[CallbackManager, List[CallbackHandler]]] = None,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.memory = memory
        if isinstance(tools, ToolManager):
            self._tool_manager = tools
        else:
            self._tool_manager = ToolManager(tools)
        if callbacks is None:
            callbacks = get_default_callbacks()
        if isinstance(callbacks, CallbackManager):
            self._callback_manager = callbacks
        else:
            self._callback_manager = CallbackManager(callbacks)

    async def async_run(self, prompt: str) -> AgentResponse:
        await self._callback_manager.on_agent_start(agent=self, prompt=prompt)
        agent_resp = await self._async_run(prompt)
        await self._callback_manager.on_agent_end(agent=self, response=agent_resp)
        return agent_resp

    def import_tool(self, tool: Tool) -> None:
        self._tool_manager.add_tool(tool)

    def reset(self) -> None:
        self.memory.clear_chat_history()

    @abc.abstractmethod
    async def _async_run(self, prompt: str) -> AgentResponse:
        raise NotImplementedError

    async def _async_run_tool(self, tool_name: str, tool_args: str) -> str:
        tool = self._tool_manager.get_tool(tool_name)
        await self._callback_manager.on_tool_start(agent=self, tool=tool, input_args=tool_args)
        try:
            tool_resp = await self._async_run_tool_without_hooks(tool, tool_args)
        except (Exception, KeyboardInterrupt) as e:
            await self._callback_manager.on_tool_error(agent=self, tool=tool, error=e)
            raise
        await self._callback_manager.on_tool_end(agent=self, tool=tool, response=tool_resp)
        return tool_resp

    async def _async_run_llm(self, messages: List[Message], **opts: Any) -> AIMessage:
        await self._callback_manager.on_llm_start(agent=self, llm=self.llm, messages=messages)
        try:
            llm_resp = await self._async_run_llm_without_hooks(messages, **opts)
        except (Exception, KeyboardInterrupt) as e:
            await self._callback_manager.on_llm_error(agent=self, llm=self.llm, error=e)
            raise
        await self._callback_manager.on_llm_end(agent=self, llm=self.llm, response=llm_resp)
        return llm_resp

    async def _async_run_tool_without_hooks(self, tool: Tool, tool_args: str) -> str:
        bnd_args = self._parse_tool_args(tool, tool_args)
        tool_resp = await tool.async_run(*bnd_args.args, **bnd_args.kwargs)
        tool_resp = json.dumps(tool_resp)
        return tool_resp

    async def _async_run_llm_without_hooks(
        self, messages: List[Message], functions=None, **opts: Any
    ) -> AIMessage:
        llm_resp = await self.llm.async_chat(messages, functions=functions, stream=False, **opts)
        return llm_resp

    def _parse_tool_args(self, tool: Tool, tool_args: str) -> inspect.BoundArguments:
        args_dict = json.loads(tool_args)
        if not isinstance(args_dict, dict):
            raise ValueError("`tool_args` cannot be interpreted as a dict.")
        # TODO: Check types
        sig = inspect.signature(tool.async_run)
        bnd_args = sig.bind(**args_dict)
        bnd_args.apply_defaults()
        return bnd_args
