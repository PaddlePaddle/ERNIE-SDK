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

import json
import logging
from collections import deque
from dataclasses import dataclass, replace
from typing import Deque, List, Optional, Tuple, Union

from erniebot_agent.agents.agent import Agent
from erniebot_agent.agents.callback.callback_manager import CallbackManager
from erniebot_agent.agents.callback.handlers.base import CallbackHandler
from erniebot_agent.agents.schema import (
    NO_ACTION_STEP,
    AgentResponse,
    AgentStep,
    NoActionStep,
    PluginStep,
    ToolInfo,
    ToolResponse,
    ToolStep,
)
from erniebot_agent.chat_models.erniebot import BaseERNIEBot
from erniebot_agent.file.base import File
from erniebot_agent.file.file_manager import FileManager
from erniebot_agent.memory import Memory
from erniebot_agent.memory.messages import (
    AIMessage,
    FunctionCall,
    FunctionMessage,
    HumanMessage,
    Message,
    SystemMessage,
)
from erniebot_agent.tools.base import BaseTool
from erniebot_agent.tools.tool_manager import ToolManager

_MAX_STEPS = 5
_logger = logging.getLogger(__name__)


class FunctionAgent(Agent):
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

    llm: BaseERNIEBot
    memory: Memory
    max_steps: int

    def __init__(
        self,
        llm: BaseERNIEBot,
        tools: Union[ToolManager, List[BaseTool]],
        *,
        memory: Optional[Memory] = None,
        system_message: Optional[SystemMessage] = None,
        callbacks: Optional[Union[CallbackManager, List[CallbackHandler]]] = None,
        file_manager: Optional[FileManager] = None,
        plugins: Optional[List[str]] = None,
        max_steps: Optional[int] = None,
        first_tools: Optional[List[BaseTool]] = [],
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
            first_tools: A list of tools that will be used first when running
                the agent.
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
        if first_tools:
            self._first_tools = first_tools
            for tool in self._first_tools:
                if tool not in self.get_tools():
                    raise RuntimeError("The first tool must be in the tools list.")
        else:
            self._first_tools = []
        self._snapshots: Deque[FunctionAgentRunSnapshot] = deque(maxlen=5)
        self._snapshot_for_curr_run: Optional[FunctionAgentRunSnapshot] = None

    @property
    def snapshots(self) -> Deque["FunctionAgentRunSnapshot"]:
        return self._snapshots

    def restore_from_snapshot(self, snapshot: "FunctionAgentRunSnapshot") -> None:
        self._snapshot_for_curr_run = snapshot

    async def _run(self, prompt: str, files: Optional[List[File]] = None) -> AgentResponse:
        chat_history: List[Message] = []
        steps_taken: List[AgentStep] = []

        if self._snapshot_for_curr_run is not None:
            chat_history[:] = self._snapshot_for_curr_run.chat_history
            steps_taken[:] = self._snapshot_for_curr_run.steps
            num_steps_taken = len(steps_taken)
            self._snapshot_for_curr_run = None
        else:
            run_input = await HumanMessage.create_with_files(
                prompt, files or [], include_file_urls=self.file_needs_url
            )
            chat_history.append(run_input)

        for tool in self._first_tools:
            curr_step, new_messages = await self._step(chat_history, use_tool=tool)
            chat_history.extend(new_messages)
            num_steps_taken += 1
            if not isinstance(curr_step, NoActionStep):
                steps_taken.append(curr_step)
            else:
                # If tool choice not work, skip this round
                _logger.warning(f"Selected tool [{tool.tool_name}] not work")
                chat_history.pop()

        while num_steps_taken < self.max_steps:
            curr_step, new_messages = await self._step(chat_history)
            chat_history.extend(new_messages)
            if not isinstance(curr_step, NoActionStep):
                steps_taken.append(curr_step)

            if isinstance(curr_step, (NoActionStep, PluginStep)):  # plugin with action
                response = self._create_finished_response(chat_history, steps_taken)
                self.memory.add_message(chat_history[0])
                self.memory.add_message(chat_history[-1])
                return response
            self._take_snapshot(chat_history, steps_taken)
            num_steps_taken += 1
        response = self._create_stopped_response(chat_history, steps_taken)
        return response

    async def _step(
        self, chat_history: List[Message], use_tool: Optional[BaseTool] = None
    ) -> Tuple[AgentStep, List[Message]]:
        new_messages: List[Message] = []
        input_messages = self.memory.get_messages() + chat_history
        if use_tool is not None:
            tool_choice = {"type": "function", "function": {"name": use_tool.tool_name}}
            llm_resp = await self.run_llm(
                messages=input_messages,
                functions=[use_tool.function_call_schema()],  # only regist one tool
                system=self.system_message.content if self.system_message is not None else None,
                plugins=self._plugins,
                tool_choice=tool_choice,
            )
        else:
            llm_resp = await self.run_llm(
                messages=input_messages,
                functions=self._tool_manager.get_tool_schemas(),
                system=self.system_message.content if self.system_message is not None else None,
                plugins=self._plugins,
            )
        output_message = llm_resp.message  # AIMessage
        new_messages.append(output_message)
        if output_message.function_call is not None:
            tool_name = output_message.function_call["name"]
            tool_args = output_message.function_call["arguments"]
            tool_resp = await self.run_tool(tool_name=tool_name, tool_args=tool_args)
            new_messages.append(FunctionMessage(name=tool_name, content=tool_resp.json))
            return (
                ToolStep(
                    info=ToolInfo(tool_name=tool_name, tool_args=tool_args),
                    result=tool_resp.json,
                    input_files=tool_resp.input_files,
                    output_files=tool_resp.output_files,
                ),
                new_messages,
            )
        elif output_message.plugin_info is not None:
            file_manager = await self.get_file_manager()
            return (
                PluginStep(
                    info=output_message.plugin_info,
                    result=output_message.content,
                    input_files=file_manager.sniff_and_extract_files_from_text(
                        chat_history[-1].content
                    ),  # TODO: make sure this is correct.
                    output_files=file_manager.sniff_and_extract_files_from_text(output_message.content),
                ),
                new_messages,
            )
        else:
            return NO_ACTION_STEP, new_messages

    def _create_finished_response(
        self,
        chat_history: List[Message],
        steps: List[AgentStep],
    ) -> AgentResponse:
        last_message = chat_history[-1]
        return AgentResponse(
            text=last_message.content,
            chat_history=chat_history,
            steps=steps,
            status="FINISHED",
        )

    def _create_stopped_response(
        self,
        chat_history: List[Message],
        steps: List[AgentStep],
    ) -> AgentResponse:
        return AgentResponse(
            text="Agent run stopped early.",
            chat_history=chat_history,
            steps=steps,
            status="STOPPED",
        )

    def _take_snapshot(self, chat_history: List[Message], steps: List[AgentStep]) -> None:
        snapshot = FunctionAgentRunSnapshot(chat_history=chat_history.copy(), steps=steps.copy())
        self._snapshots.append(snapshot)


@dataclass
class FunctionAgentRunSnapshot(object):
    chat_history: List[Message]
    steps: List[AgentStep]

    def update_last_tool_call(self, tool_name: str, tool_args: str, tool_resp: ToolResponse) -> None:
        tool_step = self.steps[-1]
        if not isinstance(tool_step, ToolStep):
            raise RuntimeError("The last step is not a tool step.")
        ai_message = self.chat_history[-2]
        function_message = self.chat_history[-1]
        if (
            not isinstance(ai_message, AIMessage)
            or ai_message.function_call is None
            or not isinstance(function_message, FunctionMessage)
        ):
            raise RuntimeError(
                "The last two messages must be "
                " an AI message containing a function call and a function message."
            )

        new_ai_message = AIMessage(
            content="",
            function_call=FunctionCall(
                name=tool_name,
                thoughts=f"我需要调用{tool_name}解决这个问题",
                arguments=tool_args,
            ),
        )
        new_function_message = FunctionMessage(name=tool_name, content=tool_resp.json)
        new_tool_step = ToolStep(
            info=ToolInfo(tool_name=tool_name, tool_args=tool_args),
            result=tool_resp.json,
            input_files=tool_resp.input_files,
            output_files=tool_resp.output_files,
        )

        self.steps.pop()
        self.steps.append(new_tool_step)
        self.chat_history.pop()
        self.chat_history.pop()
        self.chat_history.append(new_ai_message)
        self.chat_history.append(new_function_message)

    def update_last_tool_prompt(self, prompt: str) -> None:
        tool_step = self.steps[-1]
        if not isinstance(tool_step, ToolStep):
            raise RuntimeError("The last step is not a tool step.")
        function_message = self.chat_history[-1]
        if not isinstance(function_message, FunctionMessage):
            raise RuntimeError("The last message is not a function message.")

        tool_ret_json = tool_step.result
        tool_ret = json.loads(tool_ret_json)
        if isinstance(tool_ret, dict):
            raise ValueError("The return value of the tool could not be converted to a dict.")
        tool_ret["prompt"] = prompt
        updated_tool_ret_json = json.dumps(tool_ret, ensure_ascii=False)

        new_function_message = FunctionMessage(name=function_message.name, content=updated_tool_ret_json)
        new_tool_step = replace(tool_step, result=updated_tool_ret_json)

        self.steps.pop()
        self.steps.append(new_tool_step)
        self.chat_history.pop()
        self.chat_history.append(new_function_message)
