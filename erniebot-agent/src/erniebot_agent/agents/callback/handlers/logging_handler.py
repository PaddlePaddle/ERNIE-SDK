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
from typing import TYPE_CHECKING, List, Optional, Union

from erniebot_agent.agents.callback.handlers.base import CallbackHandler
from erniebot_agent.agents.schema import AgentResponse, LLMResponse, ToolResponse
from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.memory.messages import Message
from erniebot_agent.tools.base import BaseTool
from erniebot_agent.utils.json import to_pretty_json
from erniebot_agent.utils.output_style import ColoredContent

default_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from erniebot_agent.agents.base import Agent


class LoggingHandler(CallbackHandler):
    logger: logging.Logger

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__()

        if logger is None:
            self.logger = default_logger
        else:
            self.logger = logger

    async def on_run_start(self, agent: Agent, prompt: str) -> None:
        self.agent_info(
            "%s is about to start running with input:\n%s",
            agent.__class__.__name__,
            ColoredContent(prompt, role="user"),
            subject="Run",
            state="Start",
        )

    async def on_llm_start(self, agent: Agent, llm: ChatModel, messages: List[Message]) -> None:
        # TODO: Prettier messages
        self.agent_info(
            "%s is about to start running with input:\n%s",
            llm.__class__.__name__,
            ColoredContent(messages),
            subject="LLM",
            state="Start",
        )

    async def on_llm_end(self, agent: Agent, llm: ChatModel, response: LLMResponse) -> None:
        self.agent_info(
            "%s finished running with output:\n%s",
            llm.__class__.__name__,
            ColoredContent(response.message),
            subject="LLM",
            state="End",
        )

    async def on_llm_error(
        self, agent: Agent, llm: ChatModel, error: Union[Exception, KeyboardInterrupt]
    ) -> None:
        pass

    async def on_tool_start(self, agent: Agent, tool: BaseTool, input_args: str) -> None:
        js_inputs = to_pretty_json(input_args, from_json=True)
        self.agent_info(
            "%s is about to start running with input:\n%s",
            ColoredContent(tool.__class__.__name__, role="function"),
            ColoredContent(js_inputs, role="function"),
            subject="Tool",
            state="Start",
        )

    async def on_tool_end(self, agent: Agent, tool: BaseTool, response: ToolResponse) -> None:
        js_inputs = to_pretty_json(response.json, from_json=True)
        self.agent_info(
            "%s finished running with output:\n%s",
            ColoredContent(tool.__class__.__name__, role="function"),
            ColoredContent(js_inputs, role="function"),
            subject="Tool",
            state="End",
        )

    async def on_tool_error(
        self, agent: Agent, tool: BaseTool, error: Union[Exception, KeyboardInterrupt]
    ) -> None:
        pass

    async def on_run_end(self, agent: Agent, response: AgentResponse) -> None:
        self.agent_info("%s finished running.", agent.__class__.__name__, subject="Run", state="End")

    def agent_info(self, msg: str, *args, subject, state, **kwargs) -> None:
        msg = f"[{subject}][{state}] {msg}"
        self.logger.info(msg, *args, **kwargs)

    def agent_error(self, error: Union[Exception, KeyboardInterrupt], *args, subject, **kwargs) -> None:
        error_msg = f"[{subject}][ERROR] {error}"
        self.logger.error(error_msg, *args, **kwargs)

    def open_role_color(self, open: bool = True):
        """
        Open or close color role in log, if open, different role will have different color.

        Args:
            open (bool, optional): whether or not to open. Defaults to True.
        """
        if open:
            self.role_color = {"user": "Blue", "function": "Purple", "assistant": "Yellow"}
        else:
            self.role_color = {}
