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

from typing import List, Union

from erniebot_agent.agents.base import BaseAgent
from erniebot_agent.agents.schema import AgentResponse, LLMResponse, ToolResponse
from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.memory.messages import Message
from erniebot_agent.tools.base import BaseTool


class CallbackHandler(object):
    """The base class for callback handlers."""

    async def on_run_start(self, agent: BaseAgent, prompt: str) -> None:
        """Called when the agent starts running.

        Args:
            agent: The agent that is running.
            prompt: The prompt that the agent uses as input.
        """

    async def on_llm_start(self, agent: BaseAgent, llm: ChatModel, messages: List[Message]) -> None:
        """Called when the LLM starts running.

        Args:
            agent: The agent that is running.
            llm: The LLM that is running.
            messages: The messages that the LLM uses as input.
        """

    async def on_llm_end(self, agent: BaseAgent, llm: ChatModel, response: LLMResponse) -> None:
        """Called when the LLM ends running.

        Args:
            agent: The agent that is running.
            llm: The LLM that is running.
            response: The response that the LLM returns.
        """

    async def on_llm_error(
        self, agent: BaseAgent, llm: ChatModel, error: Union[Exception, KeyboardInterrupt]
    ) -> None:
        """Called when the LLM errors.

        Args:
            agent: The agent that is running.
            llm: The LLM that is running.
            error: The error that occured.
        """

    async def on_tool_start(self, agent: BaseAgent, tool: BaseTool, input_args: str) -> None:
        """Called when a tool starts running.

        Args:
            agent: The agent that is running.
            tool: The tool that is running.
            input_args: The input arguments that the tool uses.
        """

    async def on_tool_end(self, agent: BaseAgent, tool: BaseTool, response: ToolResponse) -> None:
        """Called when a tool ends running.

        Args:
            agent: The agent that is running.
            tool: The tool that is running.
            response: The response that the tool returns.
        """

    async def on_tool_error(
        self, agent: BaseAgent, tool: BaseTool, error: Union[Exception, KeyboardInterrupt]
    ) -> None:
        """Called when a tool errors.

        Args:
            agent: The agent that is running.
            tool: The tool that is running.
            error: The error that occured.
        """

    async def on_run_end(self, agent: BaseAgent, response: AgentResponse) -> None:
        """Called when the agent ends running.

        Args:
            agent: The agent that is running.
            response: The response that the agent returns.
        """
