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

from typing import TYPE_CHECKING, List, Union

from erniebot_agent.agents.callback.handlers.base import CallbackHandler
from erniebot_agent.agents.schema import AgentResponse
from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.messages import Message
from erniebot_agent.tools.base import Tool
from erniebot_agent.utils.json import to_pretty_json

if TYPE_CHECKING:
    from erniebot_agent.agents.base import Agent


class LoggingHandler(CallbackHandler):
    async def on_agent_start(self, agent: Agent, prompt: str) -> None:
        print(f"[agent][start] Agent {agent} starts running with input: {prompt}")

    async def on_llm_start(self, agent: Agent, llm: ChatModel, messages: List[Message]) -> None:
        print(f"[llm][start] LLM {llm} starts running with input: {messages}")

    async def on_llm_end(self, agent: Agent, llm: ChatModel, response: Message) -> None:
        print(f"[llm][end] LLM {llm} finished running with output: {response}")

    async def on_llm_error(
        self, agent: Agent, llm: ChatModel, error: Union[Exception, KeyboardInterrupt]
    ) -> None:
        pass

    async def on_tool_start(self, agent: Agent, tool: Tool, input_args: str) -> None:
        print(
            f"[tool][start] Tool {tool} starts running with input: "
            f"\n{to_pretty_json(input_args, from_json=True)}"
        )

    async def on_tool_end(self, agent: Agent, tool: Tool, response: str) -> None:
        print(
            f"[tool][end] Tool {tool} finished running with output: "
            f"\n{to_pretty_json(response, from_json=True)}",
        )

    async def on_tool_error(
        self, agent: Agent, tool: Tool, error: Union[Exception, KeyboardInterrupt]
    ) -> None:
        pass

    async def on_agent_end(self, agent: Agent, response: AgentResponse) -> None:
        print(f"[agent][end] Agent {agent} finished running with output: {response}")
