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

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

from erniebot_agent.agents.schema import AgentResponse, LLMResponse, ToolResponse
from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.file import File, FileManager
from erniebot_agent.memory import Memory
from erniebot_agent.memory.messages import Message
from erniebot_agent.tools.base import BaseTool

LLMT = TypeVar("LLMT", bound=ChatModel)


class BaseAgent(Protocol[LLMT]):
    llm: LLMT
    memory: Memory

    async def run(self, prompt: str, files: Optional[Sequence[File]] = None) -> AgentResponse:
        ...

    async def run_llm(
        self, messages: List[Message], *, use_memory: bool = False, llm_opts: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        ...

    async def run_tool(self, tool_name: str, tool_args: str) -> ToolResponse:
        ...

    def load_tool(self, tool: BaseTool) -> None:
        ...

    def unload_tool(self, tool: BaseTool) -> None:
        ...

    def get_tool(self, tool_name: str) -> BaseTool:
        ...

    def get_tools(self) -> List[BaseTool]:
        ...

    def reset_memory(self) -> None:
        ...

    def get_file_manager(self) -> FileManager:
        ...


@runtime_checkable
class AgentLike(Protocol):
    async def run(self, prompt: str, files: Optional[Sequence[File]] = None) -> AgentResponse:
        ...
