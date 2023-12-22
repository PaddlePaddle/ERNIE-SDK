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

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union

from erniebot_agent.file.base import File
from erniebot_agent.file.protocol import extract_file_ids
from erniebot_agent.memory import AIMessage, Message


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

    @functools.cached_property  # lazy and prevent extra fime from multiple calls
    def annotations(self) -> Dict[str, List]:
        annotations = self.output_dict()

        return annotations

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

    def output_dict(self) -> Dict[str, List]:
        # 1. split the text into parts and add file id to each part
        file_ids = extract_file_ids(self.text)

        places = []
        for file_id in file_ids:
            # remote file-id & local file-id may have different length.
            # TODO(shiyutang): in case of multiple same file_id
            places.append((self.text.index(file_id), len(file_id)))
        else:
            sorted(places, key=lambda x: x[0])

        split_text_list = []
        prev_idx = 0
        for place in places:
            file_start_index, file_len = place
            split_text_list.append(self.text[prev_idx:file_start_index])
            split_text_list.append(self.text[file_start_index : file_start_index + file_len])
            prev_idx = file_start_index + file_len
        else:
            split_text_list.append(self.text[prev_idx:])

        # 2. parse text to dict
        output_dict: Dict = {"content_parts": []}

        for data in split_text_list:
            if data in file_ids:
                file_object = None
                for agent_file in self.files:
                    if data == agent_file.file.id:
                        file_object = agent_file.file
                        break

                if file_object is not None:
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
