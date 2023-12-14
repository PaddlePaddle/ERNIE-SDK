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
import json
from typing import Any, Dict, List, Literal, Optional, Union

from erniebot_agent import file_io
from erniebot_agent.agents.callback.callback_manager import CallbackManager
from erniebot_agent.agents.callback.default import get_default_callbacks
from erniebot_agent.agents.callback.handlers.base import CallbackHandler
from erniebot_agent.agents.schema import (
    AgentFile,
    AgentResponse,
    LLMResponse,
    ToolResponse,
)
from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.file_io.base import File
from erniebot_agent.file_io.file_manager import FileManager
from erniebot_agent.file_io.protocol import is_local_file_id, is_remote_file_id
from erniebot_agent.memory.base import Memory
from erniebot_agent.messages import Message, SystemMessage
from erniebot_agent.tools.base import BaseTool
from erniebot_agent.tools.tool_manager import ToolManager
from erniebot_agent.utils.gradio_mixin import GradioMixin
from erniebot_agent.utils.logging import logger


class BaseAgent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    async def async_run(self, prompt: str, files: Optional[List[File]] = None) -> AgentResponse:
        raise NotImplementedError


class Agent(GradioMixin, BaseAgent):
    def __init__(
        self,
        llm: ChatModel,
        tools: Union[ToolManager, List[BaseTool]],
        memory: Memory,
        *,
        system_message: Optional[SystemMessage] = None,
        callbacks: Optional[Union[CallbackManager, List[CallbackHandler]]] = None,
        file_manager: Optional[FileManager] = None,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.memory = memory
        if isinstance(tools, ToolManager):
            self._tool_manager = tools
        else:
            self._tool_manager = ToolManager(tools)
        # 1. Get system message exist in memory
        # OR 2. overwrite by the system_message paased in the Agent.
        if system_message:
            self.system_message = system_message
        else:
            self.system_message = memory.get_system_message()
        if callbacks is None:
            callbacks = get_default_callbacks()
        if isinstance(callbacks, CallbackManager):
            self._callback_manager = callbacks
        else:
            self._callback_manager = CallbackManager(callbacks)
        if file_manager is None:
            file_manager = file_io.get_file_manager()
        self._file_manager = file_manager

    @property
    def tools(self) -> List[BaseTool]:
        return self._tool_manager.get_tools()

    async def async_run(self, prompt: str, files: Optional[List[File]] = None) -> AgentResponse:
        await self._callback_manager.on_run_start(agent=self, prompt=prompt)
        agent_resp = await self._async_run(prompt, files)
        await self._callback_manager.on_run_end(agent=self, response=agent_resp)
        return agent_resp

    def load_tool(self, tool: BaseTool) -> None:
        self._tool_manager.add_tool(tool)

    def unload_tool(self, tool: BaseTool) -> None:
        self._tool_manager.remove_tool(tool)

    def reset_memory(self) -> None:
        self.memory.clear_chat_history()

    @abc.abstractmethod
    async def _async_run(self, prompt: str, files: Optional[List[File]] = None) -> AgentResponse:
        raise NotImplementedError

    async def _async_run_tool(self, tool_name: str, tool_args: str) -> ToolResponse:
        tool = self._tool_manager.get_tool(tool_name)
        await self._callback_manager.on_tool_start(agent=self, tool=tool, input_args=tool_args)
        try:
            tool_resp = await self._async_run_tool_without_hooks(tool, tool_args)
        except (Exception, KeyboardInterrupt) as e:
            await self._callback_manager.on_tool_error(agent=self, tool=tool, error=e)
            raise
        await self._callback_manager.on_tool_end(agent=self, tool=tool, response=tool_resp)
        return tool_resp

    async def _async_run_llm(self, messages: List[Message], **opts: Any) -> LLMResponse:
        await self._callback_manager.on_llm_start(agent=self, llm=self.llm, messages=messages)
        try:
            llm_resp = await self._async_run_llm_without_hooks(messages, **opts)
        except (Exception, KeyboardInterrupt) as e:
            await self._callback_manager.on_llm_error(agent=self, llm=self.llm, error=e)
            raise
        await self._callback_manager.on_llm_end(agent=self, llm=self.llm, response=llm_resp)
        return llm_resp

    async def _async_run_tool_without_hooks(self, tool: BaseTool, tool_args: str) -> ToolResponse:
        parsed_tool_args = self._parse_tool_args(tool_args)
        # XXX: Sniffing is less efficient and probably unnecessary.
        # Can we make a protocol to statically recognize file inputs and outputs
        # or can we have the tools introspect about this?
        input_files = await self._sniff_and_extract_files_from_args(parsed_tool_args, tool, "input")
        tool_ret = await tool(**parsed_tool_args)
        if isinstance(tool_ret, dict):
            output_files = await self._sniff_and_extract_files_from_args(tool_ret, tool, "output")
        else:
            output_files = []
        tool_ret_json = json.dumps(tool_ret, ensure_ascii=False)
        return ToolResponse(json=tool_ret_json, files=input_files + output_files)

    async def _async_run_llm_without_hooks(
        self, messages: List[Message], functions=None, **opts: Any
    ) -> LLMResponse:
        llm_ret = await self.llm.async_chat(messages, functions=functions, stream=False, **opts)
        return LLMResponse(message=llm_ret)

    def _parse_tool_args(self, tool_args: str) -> Dict[str, Any]:
        try:
            args_dict = json.loads(tool_args)
        except json.JSONDecodeError:
            raise ValueError(f"`tool_args` cannot be parsed as JSON. `tool_args` is {tool_args}")

        if not isinstance(args_dict, dict):
            raise ValueError(f"`tool_args` cannot be interpreted as a dict. It loads as {args_dict} ")
        return args_dict

    async def _sniff_and_extract_files_from_args(
        self, args: Dict[str, Any], tool: BaseTool, file_type: Literal["input", "output"]
    ) -> List[AgentFile]:
        agent_files: List[AgentFile] = []
        for val in args.values():
            if isinstance(val, str):
                if is_local_file_id(val):
                    if self._file_manager is None:
                        logger.warning(
                            f"A file is used by {repr(tool)}, but the agent has no file manager to fetch it."
                        )
                        continue
                    file = self._file_manager.look_up_file_by_id(val)
                    if file is None:
                        raise RuntimeError(f"Unregistered ID {repr(val)} is used by {repr(tool)}.")
                elif is_remote_file_id(val):
                    if self._file_manager is None:
                        logger.warning(
                            f"A file is used by {repr(tool)}, but the agent has no file manager to fetch it."
                        )
                        continue
                    file = self._file_manager.look_up_file_by_id(val)
                    if file is None:
                        file = await self._file_manager.retrieve_remote_file_by_id(val)
                else:
                    continue
                agent_files.append(AgentFile(file=file, type=file_type, used_by=tool.tool_name))
            elif isinstance(val, dict):
                agent_files.extend(await self._sniff_and_extract_files_from_args(val, tool, file_type))
            elif isinstance(val, list) and isinstance(val[0], dict):
                for item in val:
                    agent_files.extend(await self._sniff_and_extract_files_from_args(item, tool, file_type))
        return agent_files
