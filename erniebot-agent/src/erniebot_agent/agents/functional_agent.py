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
from erniebot_agent.agents.schema import AgentAction, AgentFile, AgentResponse
from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.file.base import File
from erniebot_agent.file.file_manager import FileManager
from erniebot_agent.memory.messages import FunctionMessage, HumanMessage, Message, SystemMessage
from erniebot_agent.memory import Memory
from erniebot_agent.tools.base import BaseTool

_MAX_STEPS = 5


class FunctionalAgent(Agent):
    def __init__(
        self,
        llm: ChatModel,
        tools: Union[ToolManager, List[BaseTool]],
        memory: Memory,
        system_message: Optional[SystemMessage] = None,
        *,
        callbacks: Optional[Union[CallbackManager, List[CallbackHandler]]] = None,
        file_manager: Optional[FileManager] = None,
        plugins: Optional[List[str]] = None,  # None is not assigned, [] is no plugins.
        max_steps: Optional[int] = None,
    ) -> None:
        super().__init__(
            llm=llm,
            tools=tools,
            memory=memory,
            system_message=system_message,
            callbacks=callbacks,
            file_manager=file_manager,
            plugins=plugins,
        )
        if max_steps is not None:
            if max_steps <= 0:
                raise ValueError("Invalid `max_steps` value")
            self.max_steps = max_steps
        else:
            self.max_steps = _MAX_STEPS

    async def _async_run(self, prompt: str, files: Optional[List[File]] = None) -> AgentResponse:
        chat_history: List[Message] = []
        actions_taken: List[AgentAction] = []
        files_involved: List[AgentFile] = []

        run_input = await HumanMessage.create_with_files(
            prompt, files or [], include_file_urls=self.file_needs_url
        )

        num_steps_taken = 0
        next_step_input: Message = run_input
        while num_steps_taken < self.max_steps:
            curr_step_output = await self._async_step(
                next_step_input,
                chat_history,
                actions_taken,
                files_involved,
            )
            if curr_step_output is None:
                response = self._create_finished_response(chat_history, actions_taken, files_involved)
                self.memory.add_message(chat_history[0])
                self.memory.add_message(chat_history[-1])
                return response
            num_steps_taken += 1
            next_step_input = curr_step_output
        response = self._create_stopped_response(chat_history, actions_taken, files_involved)
        return response

    async def _async_step(
        self,
        step_input,
        chat_history: List[Message],
        actions: List[AgentAction],
        files: List[AgentFile],
    ) -> Optional[Message]:
        # TODO（shiyutang）: 传出插件调用信息，+callback
        maybe_action = await self._async_plan(step_input, chat_history)
        if isinstance(maybe_action, AgentAction):
            action: AgentAction = maybe_action
            tool_resp = await self._async_run_tool(tool_name=action.tool_name, tool_args=action.tool_args)
            actions.append(action)
            files.extend(tool_resp.files)
            return FunctionMessage(name=action.tool_name, content=tool_resp.json)
        else:
            return None

    async def _async_plan(
        self, input_message: Message, chat_history: List[Message]
    ) -> Optional[AgentAction]:
        chat_history.append(input_message)

        messages = self.memory.get_messages() + chat_history
        llm_resp = await self._async_run_llm(
            messages=messages,
            functions=self._tool_manager.get_tool_schemas(),
            system=self.system_message.content if self.system_message is not None else None,
            plugins=self.plugins,
        )
        output_message = llm_resp.message
        chat_history.append(output_message)
        if output_message.function_call is not None:
            return AgentAction(
                tool_name=output_message.function_call["name"],
                tool_args=output_message.function_call["arguments"],
            )
        else:
            return None

    def _create_finished_response(
        self,
        chat_history: List[Message],
        actions: List[AgentAction],
        files: List[AgentFile],
    ) -> AgentResponse:
        last_message = chat_history[-1]
        return AgentResponse(
            text=last_message.content,
            chat_history=chat_history,
            actions=actions,
            files=files,
            status="FINISHED",
        )

    def _create_stopped_response(
        self,
        chat_history: List[Message],
        actions: List[AgentAction],
        files: List[AgentFile],
    ) -> AgentResponse:
        return AgentResponse(
            text="Agent run stopped early.",
            chat_history=chat_history,
            actions=actions,
            files=files,
            status="STOPPED",
        )
