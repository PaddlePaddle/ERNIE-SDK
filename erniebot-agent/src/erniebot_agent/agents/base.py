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

from typing import Any, List, Optional, Protocol, TypeVar, runtime_checkable

from erniebot_agent.agents.schema import AgentResponse, LLMResponse, ToolResponse
from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.file.base import File
from erniebot_agent.file.file_manager import FileManager
from erniebot_agent.memory import Memory
from erniebot_agent.memory.messages import Message
from erniebot_agent.tools.base import BaseTool

LLMT = TypeVar("LLMT", bound=ChatModel)


class BaseAgent(Protocol[LLMT]):
    llm: LLMT
    memory: Memory

    async def async_run(self, prompt: str, files: Optional[List[File]] = None) -> AgentResponse:
        ...

    def load_tool(self, tool: BaseTool) -> None:
        ...

    def unload_tool(self, tool: BaseTool) -> None:
        ...

    def get_tools(self) -> List[BaseTool]:
        ...

    def reset_memory(self) -> None:
        ...

    async def _async_run_tool(self, tool_name: str, tool_args: str) -> ToolResponse:
        ...

    async def _async_run_llm(self, messages: List[Message], **opts: Any) -> LLMResponse:
        ...

    async def _get_file_manager(self) -> FileManager:
        ...


@runtime_checkable
class AgentLike(Protocol):
    async def async_run(self, prompt: str, files: Optional[List[File]] = None) -> AgentResponse:
        ...
