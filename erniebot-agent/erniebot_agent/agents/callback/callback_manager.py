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
from typing import List, Any

from erniebot_agent.agents.callback.events import EventType
from erniebot_agent.agents.callback.handlers.base import CallbackHandler


class CallbackManager(object):
    def __init__(self, handlers: List[CallbackHandler]):
        super().__init__()
        self._handlers = handlers

    def add_handler(self, handler: CallbackHandler):
        if handler in self._handlers:
            raise ValueError(
                f"The callback handler {handler} is already registered.")
        self._handlers.append(handler)

    def remove_handler(self, handler):
        try:
            self._handlers.remove(handler)
        except ValueError as e:
            raise ValueError(
                f"The callback handler {handler} is not registered.") from e

    def set_handlers(self, handlers: List[CallbackHandler]):
        self._handlers = []
        for handler in handlers:
            self.add_handler(handler)

    def remove_all_handlers(self):
        self._handlers = []

    async def handle_event(self,
                           event_type: EventType,
                           *args: Any,
                           **kwargs: Any) -> None:
        callback_name = 'on_' + event_type.name
        for handler in self._handlers:
            callback = getattr(handler, callback_name, None)
            if not inspect.iscoroutinefunction(callback):
                raise TypeError(f"Callback must be a coroutine function.")
            await callback(*args, **kwargs)

    async def on_agent_start(self, agent: Agent, prompt: str) -> None:
        await self.handle_event(
            EventType.AGENT_START, agent=agent, prompt=prompt)

    async def on_llm_start(self,
                           agent: Agent,
                           llm: LLM,
                           messages: List[Message]) -> None:
        await self.handle_event(
            EventType.LLM_START, agent=agent, llm=llm, messages=messages)

    async def on_llm_end(self, agent: Agent, llm: LLM,
                         response: Message) -> None:
        await self.handle_event(
            EventType.LLM_END, agent=agent, llm=llm, response=response)

    async def on_llm_error(self,
                           agent: Agent,
                           llm: LLM,
                           error: Union[Exception, KeyboardInterrupt]) -> None:
        await self.handle_event(
            EventType.AGENT_START, agent=agent, llm=llm, error=error)

    async def on_tool_start(self, agent: Agent, tool: Tool,
                            input_args: str) -> None:
        await self.handle_event(
            EventType.AGENT_START,
            agent=agent,
            tool=tool,
            input_args=input_args)

    async def on_tool_end(self, agent: Agent, tool: Tool,
                          response: str) -> None:
        await self.handle_event(
            EventType.AGENT_START, agent=agent, tool=tool, response=response)

    async def on_tool_error(self,
                            agent: Agent,
                            tool: Tool,
                            error: Union[Exception, KeyboardInterrupt]) -> None:
        await self.handle_event(
            EventType.AGENT_START, agent=agent, tool=tool, error=error)

    async def on_agent_end(self, agent: Agent, response: str) -> None:
        await self.handle_event(
            EventType.AGENT_START, agent=agent, response=response)
