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

from typing import List, Optional, Union

from erniebot_agent.agents.base import Agent, ToolManager
from erniebot_agent.agents.callback.callback_manager import CallbackManager
from erniebot_agent.agents.callback.handlers.base import CallbackHandler
from erniebot_agent.agents.schema import AgentAction, AgentResponse, AgentStep
from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.memory.base import Memory
from erniebot_agent.memory.whole_memory import WholeMemory
from erniebot_agent.messages import AIMessage, FunctionMessage, HumanMessage
from erniebot_agent.tools.base import Tool

_MAX_STEPS = 5


class FunctionalAgent(Agent):
    def __init__(
        self,
        llm: ChatModel,
        tools: Union[ToolManager, List[Tool]],
        memory: Memory,
        *,
        callbacks: Optional[Union[CallbackManager, List[CallbackHandler]]] = None,
        max_steps: Optional[int] = None,
        run_memory: Optional[Memory] = None,
    ) -> None:
        super().__init__(llm=llm, tools=tools, memory=memory, callbacks=callbacks)
        if max_steps is not None:
            if max_steps <= 0:
                raise ValueError("Invalid `max_steps` value")
            self.max_steps = max_steps
        else:
            self.max_steps = _MAX_STEPS
        self._run_memory = run_memory

    async def _async_run(self, prompt: str) -> AgentResponse:
        if self._run_memory:
            run_memory = self._run_memory
        else:
            run_memory = self._create_run_memory()
        self._copy_memory(self.memory, run_memory)

        ask = HumanMessage(content=prompt)
        step = await self._async_create_first_step(ask, run_memory)

        num_steps_taken = 0
        while num_steps_taken < self.max_steps:
            if isinstance(step, AgentAction):
                tool_resp = await self._async_run_tool(step.tool_name, step.tool_args)
                step = await self._async_create_next_step(step.tool_name, tool_resp, run_memory)
            elif isinstance(step, AgentResponse):
                self.memory.add_message(ask)
                self.memory.add_message(AIMessage(content=step.content))
                return step
            else:
                raise TypeError("Invalid type of step")
            num_steps_taken += 1
        response = self._create_stopped_response(run_memory)
        return response

    async def _async_plan(
        self, message: Union[HumanMessage, FunctionMessage], run_memory: Memory
    ) -> AgentStep:
        run_memory.add_message(message)
        llm_resp = await self._async_run_llm(
            messages=run_memory.get_messages(),
            functions=self._tool_manager.get_tool_function_inputs(),
        )
        run_memory.add_message(llm_resp)
        if llm_resp.function_call is not None:
            return AgentAction(
                tool_name=llm_resp.function_call["name"], tool_args=llm_resp.function_call["arguments"]
            )
        else:
            return AgentResponse(content=llm_resp.content, intermediate_messages=run_memory.get_messages())

    async def _async_create_first_step(self, message: HumanMessage, run_memory: Memory) -> AgentStep:
        return await self._async_plan(message, run_memory)

    async def _async_create_next_step(
        self, tool_name: str, tool_response: str, run_memory: Memory
    ) -> AgentStep:
        message = FunctionMessage(name=tool_name, content=tool_response)
        return await self._async_plan(message, run_memory)

    def _create_run_memory(self) -> Memory:
        return WholeMemory()

    def _create_stopped_response(self, run_memory: Memory) -> AgentResponse:
        return AgentResponse(
            content="Agent run stopped early.", intermediate_messages=run_memory.get_messages()
        )

    @staticmethod
    def _copy_memory(src: Memory, dst: Memory):
        dst.clear_chat_history()
        messages = src.get_messages()
        dst.add_messages(messages)
