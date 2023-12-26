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

from typing import List, NoReturn, Optional, Union

from erniebot_agent.agents.agent import Agent
from erniebot_agent.agents.callback.callback_manager import CallbackManager
from erniebot_agent.agents.callback.handlers.base import CallbackHandler
from erniebot_agent.agents.schema import AgentAction, AgentFile, AgentResponse
from erniebot_agent.chat_models.erniebot import BaseERNIEBot
from erniebot_agent.file.base import File
from erniebot_agent.file.file_manager import FileManager
from erniebot_agent.memory import Memory
from erniebot_agent.memory.messages import (
    FunctionMessage,
    HumanMessage,
    Message,
    SystemMessage,
)
from erniebot_agent.tools.base import BaseTool
from erniebot_agent.tools.tool_manager import ToolManager
from erniebot_agent.utils.exceptions import FileError

_MAX_STEPS = 5


class FunctionAgent(Agent[BaseERNIEBot]):
    """An agent driven by function calling.

    The orchestration capabilities of a function agent are powered by the
    function calling ability of LLMs. Typically, a function agent asks the LLM
    to generate a response that can be parsed into an action (e.g., calling a
    tool with given arguments), and then the agent takes that action, which
    forms an agent step. The agent repeats this process until the maximum number
    of steps is reached or the LLM considers the task finished.

    Attributes:
        llm: The LLM that the agent uses.
        memory: The message storage that keeps the chat history.
        max_steps: The maximum number of steps in each agent run.
    """

    def __init__(
        self,
        llm: BaseERNIEBot,
        tools: Union[ToolManager, List[BaseTool]],
        memory: Memory,
        *,
        system_message: Optional[SystemMessage] = None,
        callbacks: Optional[Union[CallbackManager, List[CallbackHandler]]] = None,
        file_manager: Optional[FileManager] = None,
        plugins: Optional[List[str]] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        """Initialize a function agent.

        Args:
            llm: An LLM for the agent to use.
            tools: A list of tools for the agent to use.
            memory: A memory object that equips the agent to remember chat
                history.
            system_message: A message that tells the LLM how to interpret the
                conversations. If `None`, the system message contained in
                `memory` will be used.
            callbacks: A list of callback handlers for the agent to use. If
                `None`, a default list of callbacks will be used.
            file_manager: A file manager for the agent to interact with files.
                If `None`, a global file manager that can be shared among
                different components will be implicitly created and used.
            plugins: A list of names of the plugins for the agent to use. If
                `None`, the agent will use a default list of plugins. Set
                `plugins` to `[]` to disable the use of plugins.
            max_steps: The maximum number of steps in each agent run. If `None`,
                use a default value.
        """
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

    async def _run(self, prompt: str, files: Optional[List[File]] = None) -> AgentResponse:
        if files:
            await self._ensure_managed_files(files)

        chat_history: List[Message] = []
        actions_taken: List[AgentAction] = []
        files_involved: List[AgentFile] = []

        run_input = await HumanMessage.create_with_files(
            prompt, files or [], include_file_urls=self.file_needs_url
        )

        num_steps_taken = 0
        next_step_input: Message = run_input
        while num_steps_taken < self.max_steps:
            curr_step_output = await self._step(
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

    async def _step(
        self,
        step_input,
        chat_history: List[Message],
        actions: List[AgentAction],
        files: List[AgentFile],
    ) -> Optional[Message]:
        # TODO（shiyutang）: 传出插件调用信息，+callback
        maybe_action = await self._plan(step_input, chat_history)
        if isinstance(maybe_action, AgentAction):
            action: AgentAction = maybe_action
            tool_resp = await self.run_tool(tool_name=action.tool_name, tool_args=action.tool_args)
            actions.append(action)
            files.extend(tool_resp.files)
            return FunctionMessage(name=action.tool_name, content=tool_resp.json)
        else:
            return None

    async def _plan(self, input_message: Message, chat_history: List[Message]) -> Optional[AgentAction]:
        chat_history.append(input_message)

        messages = self._memory.get_messages() + chat_history
        llm_resp = await self.run_llm(
            messages=messages,
            functions=self._tool_manager.get_tool_schemas(),
            system=self.system_message.content if self.system_message is not None else None,
            plugins=self._plugins,
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

    async def _ensure_managed_files(self, files: List[File]) -> None:
        def _raise_exception(file: File) -> NoReturn:
            raise FileError(f"{repr(file)} is not managed by the file manager of the agent.")

        file_manager = await self.get_file_manager()
        for file in files:
            try:
                managed_file = file_manager.look_up_file_by_id(file.id)
            except FileError:
                _raise_exception(file)
            if file is not managed_file:
                _raise_exception(file)
