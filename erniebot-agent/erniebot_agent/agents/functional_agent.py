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

from typing import Optional, Union, List

from erniebot_agent.agents.base import Agent, ToolManager
from erniebot_agent.agents.schema import AgentStep, AgentAction, AgentResponse
from erniebot_agent.agents.callback.callback_manager import CallbackManager
from erniebot_agent.agents.callback.handlers.base import CallbackHandler
from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.messages import FunctionMessage, HumanMessage, AIMessage
from erniebot_agent.tools.base import Tool

_MAX_STEPS = 50

class FunctionalAgent(Agent):
    def __init__(
        self,
        llm: ChatModel,
        tools: Union[ToolManager, List[Tool]],
        memory: Memory,
        *,
        callbacks: Optional[Union[CallbackManager, List[CallbackHandler]]] = None,
        max_steps: Optional[int]=None,
    ) -> None:
        super().__init__(llm=llm, tools=tools, memory=memory, callbacks=callbacks)
        self.max_steps = max_steps or _MAX_STEPS

    async def _run(self, prompt: str) -> str:
        # TODO: Memory when 
        temp_memory = WholeConversationMemory()
        # Copy memory
        ...
        step = await self._create_first_step(prompt, temp_memory)
        while True:
            if isinstance(step, AgentAction):
                tool_resp = await self._run_tool(step.tool_name, step.tool_args)
                step = await self._create_next_step(tool_resp, temp_memory)
            elif isinstance(step, AgentResponse):
                self.memory.add_message(AIMessage(content=step.content))
                return step.content
            else:
                raise TypeError("Invalid type of step")

    async def _plan(self, message: str, temporary_memory: Memory) -> AgentStep:
        temporary_memory.add_message(message)
        llm_resp = await self._run_llm(
            messages=temporary_memory.get_messages(),
            functions=self._tool_manager.get_tool_function_inputs(),
        )
        temporary_memory.add_message(llm_resp)
        return llm_resp

    async def _create_first_step(self, prompt: str, temporary_memory: Memory) -> AgentStep:
        message = HumanMessage(prompt)
        self.memory.add_message(message)
        return self._plan(message, temporary_memory)

    async def _create_next_step(self, tool_response: str, temporary_memory: Memory) -> AgentStep:
        message = FunctionMessage(tool_response)
        return self._plan(message, temporary_memory)
        
