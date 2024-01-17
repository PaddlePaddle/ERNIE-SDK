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
from typing import Any, Iterable, List, Optional, TypedDict, Union, final

from erniebot_agent.agents import Agent
from erniebot_agent.agents.schema import AgentResponse, AgentStep, ToolInfo, ToolStep
from erniebot_agent.chat_models.erniebot import BaseERNIEBot
from erniebot_agent.memory import AIMessage, FunctionMessage, HumanMessage, Message
from erniebot_agent.tools.base import BaseTool
from erniebot_agent.tools.tool_manager import ToolManager

_logger = logging.getLogger(__name__)


class ParseDict(TypedDict, total=False):
    steps: List[AgentStep]
    chat_history: List[Message]
    inp: Any
    result: Optional[Any]


class Edge:
    def __init__(self, next_task, condition=None):
        self.next_task = next_task
        self.condition = condition


class Task(object):
    def __init__(self, des):
        self.result_for_condition = None
        self.result = None
        self.des = des
        self.next_tasks = []
        self.end_reason = None

    def add_next_task(self, task, condition=None) -> None:
        edge = Edge(task, condition)
        self.next_tasks.append(edge)

    @final
    async def execute(self, inp, **kwargs) -> ParseDict:
        result_dict = await self._execute(inp, **kwargs)
        self._check_output(result_dict)
        return result_dict

    def _check_output(self, result_dict):
        force_key = list(ParseDict.__annotations__.keys())
        force_key.extend(["result"])

        for k in force_key:
            if k not in result_dict:
                raise ValueError(f"Task {self.__name__} must have a key named '{k}' in the result")

    async def _execute(self, inp, **kwargs) -> ParseDict:
        raise NotImplementedError("Subclasses must implement the `_execute` method")


class ManualAgent(Agent):
    def __init__(self, llm: BaseERNIEBot, tools: Union[ToolManager, Iterable[BaseTool]], **kwargs):
        super().__init__(llm=llm, tools=tools, **kwargs)
        self.tasks: Iterable[Task] = []
        self.start_task: Optional[Task] = None

    def add_task(self, task):
        if task in self.tasks:
            raise RuntimeError(f"Task {task.__name__} is already in the agent")
        self.tasks.append(task)

    async def _run(self, prompt, files):
        if self.start_task is None:
            self.set_start_task(self.tasks[0])
            _logger.warning("No start task is set, the first task will be used as the start task")
        current_task = self.start_task
        run_input = await HumanMessage.create_with_files(
            prompt, files or [], include_file_urls=self.file_needs_url
        )
        cur_inp = {"inp": run_input, "steps": [], "chat_history": []}
        while current_task:
            result = await current_task.execute(cur_inp)
            next_task = self.get_next_task(current_task, result)
            current_task = next_task
            cur_inp["inp"] = cur_inp["result"]

        response = self._create_finished_response(
            result["result"], cur_inp["chat_history"], cur_inp["steps"]
        )
        return response

    def set_start_task(self, task):
        self.start_task = task

    def get_next_task(self, current_task, result):
        for edge in current_task.next_tasks:
            if edge.condition is None or edge.condition(result):
                return edge.next_task
        return None

    def _create_finished_response(
        self,
        response: Any,
        chat_history: List[Message],
        steps: List[AgentStep],
    ) -> AgentResponse:
        text = response.content if isinstance(response, Message) else str(response)
        return AgentResponse(
            text=text,
            chat_history=chat_history,
            steps=steps,
            status="FINISHED",
        )


class AgentTask(Task):
    # input_type: Type = Union[FunctionMessage, HumanMessage]

    def __init__(self, des, agent: ManualAgent, selected_tool: Optional[BaseTool] = None):
        super().__init__(des)
        self.agent = agent
        self.selected_tool = selected_tool
        if self.selected_tool is not None:
            if self.selected_tool not in self.agent.get_tools():
                raise RuntimeError(
                    "Selected tool is not in the available tools"
                    "Please Use `agent.load_tool(tool)` to load the tool"
                )

        if self not in agent.tasks:
            agent.add_task(self)

    async def _execute(self, inp: ParseDict, **llm_opts: Any) -> ParseDict:
        if "tool_choice" in llm_opts:
            raise ValueError("`tool_choice` can not set in the input arguments")

        assert isinstance(inp["inp"], (FunctionMessage, HumanMessage))

        if self.selected_tool is not None:
            llm_opts["tool_choice"] = {
                "type": "function",
                "function": {"name": self.selected_tool.tool_name},
            }
            if "functions" in llm_opts:
                _logger.warning("`functions` in input arguments will be ignored")
            llm_opts["functions"] = [self.selected_tool.function_call_schema()]

        inp["chat_history"].append(inp["inp"])
        llm_resp = await self.agent.run_llm(inp["chat_history"], **llm_opts)
        output_message: AIMessage = llm_resp.message
        inp["chat_history"].append(output_message)

        if output_message.function_call is not None:
            tool_name = output_message.function_call["name"]
            tool_args = output_message.function_call["arguments"]
            tool_resp = await self.agent.run_tool(tool_name=tool_name, tool_args=tool_args)
            inp["steps"].append(
                ToolStep(
                    info=ToolInfo(tool_name=tool_name, tool_args=tool_args),
                    result=tool_resp.json,
                    input_files=tool_resp.input_files,
                    output_files=tool_resp.output_files,
                )
            )
            function_result = FunctionMessage(name=tool_name, content=tool_resp.json)
            inp["result"] = function_result
        else:
            inp["result"] = output_message

        return inp
