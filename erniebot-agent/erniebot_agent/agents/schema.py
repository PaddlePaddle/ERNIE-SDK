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

import asyncio
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from erniebot_agent.file_io.base import File
from erniebot_agent.file_io.file_manager import FileManager
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
    annotations: Optional[Dict[str, List]] = None

    def get_annotations(self):
        # If there is file, annotation will not be none,
        if self.includes_file():
            self.annotations = self.output_dict()
        return self.annotations

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

    def includes_file(self) -> bool:
        for agent_file in self.files:  # TODO: self.files contains all input and output files
            if agent_file.file.id in self.text:
                return True
        else:
            return False

    def output_dict(self) -> Dict[str, List]:
        # split the text into parts and add file id to each part
        places = re.finditer("file-", self.text)
        split_text_list = []
        prev_idx = 0
        file_len = len(self.files[0].file.id)
        for place in places:
            place = place.start()
            split_text_list.append(self.text[prev_idx:place])
            split_text_list.append(self.text[place : place + file_len])
            prev_idx = place + file_len
        else:
            split_text_list.append(self.text[prev_idx:])

        output_dict: Dict = {"content_parts": []}
        file_manager = FileManager()

        for data in split_text_list:
            if "file-" in data:
                if "local" in data:
                    file_object = file_manager.look_up_file_by_id(data)
                elif "remote" in data:
                    file_object = asyncio.run(file_manager.retrieve_remote_file_by_id(data))
                else:
                    raise RuntimeError(f"File id is neither local nor remote. It is {data}")

                file_meta = file_object.to_dict()

                output_dict["content_parts"].append(file_meta)
            else:
                output_dict["content_parts"].append({"text": data})

        return output_dict


@dataclass
class AgentFile(object):
    """A file that is used by an agent."""

    file: File
    type: Literal["input", "output"]
    used_by: str
