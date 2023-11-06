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

from typing import List, Union

from erniebot_agent.agents.callback.handlers.base import CallbackHandler
from erniebot_agent.agents.base import Agent
from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.messages import Message
from erniebot_agent.tools.base import Tool
from erniebot_agent.utils.logging import logger
from erniebot_agent.utils.json import to_pretty_json


class LoggingHandler(CallbackHandler):
    async def on_agent_start(self, agent: Agent, prompt: str) -> None:
        logger.info("[agent][start] Agent %s starts running with input: %s", str(agent), prompt)

    async def on_llm_start(self, agent: Agent, llm: ChatModel, messages: List[Message]) -> None:
        logger.info("[llm][start] LLM %s starts running with input: %s", str(llm), str(messages))

    async def on_llm_end(self, agent: Agent, llm: ChatModel, response: Message) -> None:
        logger.info("[llm][end] LLM %s finished running with output: %s", str(response))

    async def on_llm_error(
        self, agent: Agent, llm: ChatModel, error: Union[Exception, KeyboardInterrupt]
    ) -> None:
        pass

    async def on_tool_start(self, agent: Agent, tool: Tool, input_args: str) -> None:
        logger.info("[tool][start] Tool %s starts running with input: \n%s", str(tool), to_pretty_json(input_args, from_json=True))

    async def on_tool_end(self, agent: Agent, tool: Tool, response: str) -> None:
        logger.info("[tool][end] Tool %s finished running with output: \n%s", str(tool), to_pretty_json(response, from_json=True))

    async def on_tool_error(
        self, agent: Agent, tool: Tool, error: Union[Exception, KeyboardInterrupt]
    ) -> None:
        pass

    async def on_agent_end(self, agent: Agent, response: str) -> None:
        logger.info("[agent][end] Agent %s finished running with output: %s", str(agent), response)
