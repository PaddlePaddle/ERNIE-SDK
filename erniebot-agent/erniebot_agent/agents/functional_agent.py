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
from erniebot_agent.agents.schema import AgentAction, AgentResponse
from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.memory.base import Memory
from erniebot_agent.messages import FunctionMessage, HumanMessage, Message
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
    ) -> None:
        super().__init__(llm=llm, tools=tools, memory=memory, callbacks=callbacks)
        if max_steps is not None:
            if max_steps <= 0:
                raise ValueError("Invalid `max_steps` value")
            self.max_steps = max_steps
        else:
            self.max_steps = _MAX_STEPS

    async def _async_run(self, prompt: str) -> AgentResponse:
        chat_history: List[Message] = []
        actions_taken: List[AgentAction] = []
        ask = HumanMessage(content=prompt)

        num_steps_taken = 0
        next_step_input: Message = ask
        while num_steps_taken < self.max_steps:
            curr_step_output = await self._async_step(next_step_input, chat_history, actions_taken)
            if curr_step_output is None:
                response = self._create_finished_response(chat_history, actions_taken)
                self.memory.add_message(chat_history[0])
                self.memory.add_message(chat_history[-1])
                return response
            num_steps_taken += 1
            next_step_input = curr_step_output
        response = self._create_stopped_response(chat_history, actions_taken)
        return response

    async def _async_step(
        self, step_input, chat_history: List[Message], actions: List[AgentAction]
    ) -> Optional[Message]:
        maybe_action = await self._async_plan(step_input, chat_history)
        if isinstance(maybe_action, AgentAction):
            action: AgentAction = maybe_action
            tool_output = await self._async_run_tool(tool_name=action.tool_name, tool_args=action.tool_args)
            actions.append(action)
            return FunctionMessage(name=action.tool_name, content=tool_output)
        else:
            return None

    async def _async_plan(
        self, input_message: Message, chat_history: List[Message]
    ) -> Optional[AgentAction]:
        chat_history.append(input_message)
        messages = self.memory.get_messages() + chat_history
        output_message = await self._async_run_llm(
            messages=messages,
            functions=self._tool_manager.get_tool_schemas(),
        )
        chat_history.append(output_message)
        if output_message.function_call is not None:
            return AgentAction(
                tool_name=output_message.function_call["name"],  # type: ignore
                tool_args=output_message.function_call["arguments"],
            )
        else:
            return None

    def _create_finished_response(
        self, chat_history: List[Message], actions: List[AgentAction]
    ) -> AgentResponse:
        last_message = chat_history[-1]
        return AgentResponse(
            content=last_message.content,
            chat_history=chat_history,
            actions=actions,
            status="FINISHED",
        )

    def _create_stopped_response(
        self, chat_history: List[Message], actions: List[AgentAction]
    ) -> AgentResponse:
        return AgentResponse(
            content="Agent run stopped early.",
            chat_history=chat_history,
            actions=actions,
            status="STOPPED",
        )
