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

import logging
from typing import List, Optional

from erniebot_agent.agents.base import BaseAgent
from erniebot_agent.agents.callback.handlers.base import CallbackHandler
from erniebot_agent.agents.schema import AgentResponse, LLMResponse, ToolResponse
from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.memory.messages import Message
from erniebot_agent.tools.base import BaseTool
from erniebot_agent.utils.json import to_pretty_json
from erniebot_agent.utils.output_style import ColoredContent

default_logger = logging.getLogger(__name__)


class LoggingHandler(CallbackHandler):
    """A callback handler for logging."""

    logger: logging.Logger

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize a logging handler.

        Args:
            logger: The logger to use. If `None`, a default logger will be used.
        """
        super().__init__()

        if logger is None:
            self.logger = default_logger
        else:
            self.logger = logger

    async def on_run_start(self, agent: BaseAgent, prompt: str) -> None:
        """Called to log when the agent starts running."""
        self._agent_info(
            "%s is about to start running with input:\n%s",
            agent.__class__.__name__,
            ColoredContent(prompt, role="user"),
            subject="Run",
            state="Start",
        )

    async def on_llm_start(self, agent: BaseAgent, llm: ChatModel, messages: List[Message]) -> None:
        """Called to log when the LLM starts running."""
        # TODO: Prettier messages
        self._agent_info(
            "%s is about to start running with input:\n%s",
            llm.__class__.__name__,
            ColoredContent(messages[-1]),
            subject="LLM",
            state="Start",
        )

    async def on_llm_end(self, agent: BaseAgent, llm: ChatModel, response: LLMResponse) -> None:
        """Called to log when the LLM ends running."""
        self._agent_info(
            "%s finished running with output:\n%s",
            llm.__class__.__name__,
            ColoredContent(response.message),
            subject="LLM",
            state="End",
        )

    async def on_tool_start(self, agent: BaseAgent, tool: BaseTool, input_args: str) -> None:
        """Called to log when a tool starts running."""
        js_inputs = to_pretty_json(input_args, from_json=True)
        self._agent_info(
            "%s is about to start running with input:\n%s",
            ColoredContent(tool.__class__.__name__, role="function"),
            ColoredContent(js_inputs, role="function"),
            subject="Tool",
            state="Start",
        )

    async def on_tool_end(self, agent: BaseAgent, tool: BaseTool, response: ToolResponse) -> None:
        """Called to log when a tool ends running."""
        js_inputs = to_pretty_json(response.json, from_json=True)
        self._agent_info(
            "%s finished running with output:\n%s",
            ColoredContent(tool.__class__.__name__, role="function"),
            ColoredContent(js_inputs, role="function"),
            subject="Tool",
            state="End",
        )

    async def on_run_end(self, agent: BaseAgent, response: AgentResponse) -> None:
        """Called to log when the agent ends running."""
        self._agent_info("%s finished running.", agent.__class__.__name__, subject="Run", state="End")

    def _agent_info(self, msg: str, *args, subject, state, **kwargs) -> None:
        msg = f"[{subject}][{state}] {msg}"
        self.logger.info(msg, *args, **kwargs)
