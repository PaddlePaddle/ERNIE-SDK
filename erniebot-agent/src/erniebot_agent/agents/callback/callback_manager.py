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

import inspect
from typing import Any, List, Union, final

from erniebot_agent.agents.base import BaseAgent
from erniebot_agent.agents.callback.event import EventType
from erniebot_agent.agents.callback.handlers.base import CallbackHandler
from erniebot_agent.agents.schema import AgentResponse, LLMResponse, ToolResponse
from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.memory.messages import Message
from erniebot_agent.tools.base import BaseTool


@final
class CallbackManager(object):
    """The manager for callback handlers."""

    def __init__(self, handlers: List[CallbackHandler]):
        """Initialize a callback manager.

        Args:
            handlers: A list of callback handlers.
        """
        super().__init__()
        self._handlers: List[CallbackHandler] = []
        self.set_handlers(handlers)

    @property
    def handlers(self) -> List[CallbackHandler]:
        """The list of callback handlers."""
        return self._handlers

    def add_handler(self, handler: CallbackHandler):
        """Add a callback handler."""
        self._handlers.append(handler)

    def remove_handler(self, handler: CallbackHandler):
        """Remove a callback handler."""
        self._handlers.remove(handler)

    def set_handlers(self, handlers: List[CallbackHandler]):
        """Set the callback handlers."""
        self._handlers[:] = handlers

    def remove_all_handlers(self):
        """Remove all callback handlers."""
        self._handlers.clear()

    async def on_run_start(self, agent: BaseAgent, prompt: str, **kwargs) -> None:
        await self._handle_event(EventType.RUN_START, agent=agent, prompt=prompt, **kwargs)

    async def on_llm_start(self, agent: BaseAgent, llm: ChatModel, messages: List[Message]) -> None:
        await self._handle_event(EventType.LLM_START, agent=agent, llm=llm, messages=messages)

    async def on_llm_end(self, agent: BaseAgent, llm: ChatModel, response: LLMResponse) -> None:
        await self._handle_event(EventType.LLM_END, agent=agent, llm=llm, response=response)

    async def on_llm_error(
        self, agent: BaseAgent, llm: ChatModel, error: Union[Exception, KeyboardInterrupt]
    ) -> None:
        await self._handle_event(EventType.LLM_ERROR, agent=agent, llm=llm, error=error)

    async def on_tool_start(self, agent: BaseAgent, tool: BaseTool, input_args: str) -> None:
        await self._handle_event(EventType.TOOL_START, agent=agent, tool=tool, input_args=input_args)

    async def on_tool_end(self, agent: BaseAgent, tool: BaseTool, response: ToolResponse) -> None:
        await self._handle_event(EventType.TOOL_END, agent=agent, tool=tool, response=response)

    async def on_tool_error(
        self, agent: BaseAgent, tool: BaseTool, error: Union[Exception, KeyboardInterrupt]
    ) -> None:
        await self._handle_event(EventType.TOOL_ERROR, agent=agent, tool=tool, error=error)

    async def on_run_end(self, agent: BaseAgent, response: AgentResponse, **kwargs) -> None:
        await self._handle_event(EventType.RUN_END, agent=agent, response=response, **kwargs)

    async def _handle_event(self, event_type: EventType, *args: Any, **kwargs: Any) -> None:
        callback_name = "on_" + event_type.value
        for handler in self._handlers:
            callback = getattr(handler, callback_name, None)
            if not inspect.iscoroutinefunction(callback):
                raise RuntimeError("Callback must be a coroutine function.")
            await callback(*args, **kwargs)
