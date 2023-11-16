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

import logging
from typing import TYPE_CHECKING, List, Union

import erniebot_agent
from erniebot_agent.agents.callback.handlers.base import CallbackHandler
from erniebot_agent.agents.schema import AgentResponse
from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.messages import Message
from erniebot_agent.tools.base import Tool
from erniebot_agent.utils.json import to_pretty_json

if TYPE_CHECKING:
    from erniebot_agent.agents.base import Agent


class LoggingHandler(CallbackHandler):
    def __init__(self, logger: logging.Logger = erniebot_agent.logger) -> None:
        self.logger = logger

    async def on_run_start(self, agent: Agent, prompt: str) -> None:
        self.agent_info(
            "Agent %s starts running with input: %s",
            agent,
            prompt,
            level="Run",
            state="Start",
        )

    async def on_llm_start(self, agent: Agent, llm: ChatModel, messages: List[Message]) -> None:
        self.agent_info("Agent %s starts running with input: %s", llm, messages, level="LLM", state="Start")

    async def on_llm_end(self, agent: Agent, llm: ChatModel, response: Message) -> None:
        self.agent_info("Agent %s starts running with input: %s", llm, response, level="LLM", state="End")

    async def on_llm_error(
        self, agent: Agent, llm: ChatModel, error: Union[Exception, KeyboardInterrupt]
    ) -> None:
        self.logger.error("[LLM][Error] %s", error)

    async def on_tool_start(self, agent: Agent, tool: Tool, input_args: str) -> None:
        self.agent_info(
            "Tool %s starts running with input: \n %s",
            tool,
            to_pretty_json(input_args, from_json=True),
            level="Tool",
            state="Start",
        )

    async def on_tool_end(self, agent: Agent, tool: Tool, response: str) -> None:
        self.agent_info(
            "Tool %s finished running with input: \n %s",
            tool,
            to_pretty_json(response, from_json=True),
            level="Tool",
            state="End",
        )

    async def on_tool_error(
        self, agent: Agent, tool: Tool, error: Union[Exception, KeyboardInterrupt]
    ) -> None:
        self.logger.error("[Tool][Error] %s", error)

    async def on_run_end(self, agent: Agent, response: AgentResponse) -> None:
        self.agent_info(
            "Agent %s finished running with output: %s", agent, response, level="Run", state="End"
        )

    def agent_info(self, msg: str, *args, level="Run", state="Start", **kwargs) -> None:
        msg = f"[{level}][{state}]{msg}"
        return self.logger.info(msg, *args, **kwargs)
