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

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from erniebot_agent.file_io.base import File
from erniebot_agent.messages import AIMessage, Message
from typing_extensions import Literal


@dataclass
class AgentAction(object):
    """An action for an agent to execute."""

    tool_name: str
    tool_args: str


@dataclass
class AgentPlan(object):
    """A plan that contains a list of actions."""

    actions: List[AgentAction]


@dataclass
class LLMResponse(object):
    """A response from an LLM."""

    message: AIMessage


@dataclass
class ToolResponse(object):
    """A response from a tool."""

    json: str
    files: List["AgentFile"]


@dataclass
class AgentResponse(object):
    """The final response from an agent."""

    text: str
    chat_history: List[Message]
    actions: List[AgentAction]
    files: List["AgentFile"]
    status: Union[Literal["FINISHED"], Literal["STOPPED"]]

    def get_last_output_file(self) -> Optional[File]:
        for agent_file in self.files[::-1]:
            if agent_file.type == "output":
                return agent_file.file
        else:
            return None

    def get_output_files(self) -> List[File]:
        return [agent_file.file for agent_file in self.files if agent_file.type == "output"]

    def get_tool_input_output_files(self, tool_name: str) -> Tuple[List[File], List[File]]:
        input_files: List[File] = []
        output_files: List[File] = []
        for agent_file in self.files:
            if agent_file.used_by == tool_name:
                if agent_file.type == "input":
                    input_files.append(agent_file.file)
                elif agent_file.type == "output":
                    output_files.append(agent_file.file)
                else:
                    raise RuntimeError("File type is neither input nor output.")
        return input_files, output_files


@dataclass
class AgentFile(object):
    """A file that is used by an agent."""

    file: File
    type: Literal["input", "output"]
    used_by: str
