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

import functools
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Sequence, TypeVar, Union

from typing_extensions import Literal

from erniebot_agent.file import protocol
from erniebot_agent.file.base import File
from erniebot_agent.memory import AIMessage, Message
from erniebot_agent.memory.messages import PluginInfo


@dataclass
class ToolAction(object):
    """A tool calling action for an agent to execute."""

    tool_name: str
    tool_args: str


@dataclass
class PluginAction(object):  # save for plugins that can be planned
    """A plugin calling action for an agent to execute."""

    plugin_name: str
    finish_reason: str


class ToolInfo(Dict):
    tool_name: str
    tool_args: str


@dataclass
class NullAction(object):
    """This class indicates that no action shall be taken by the agent."""


NULL_ACTION = NullAction()


# Note: save for plugins that can be planned
AgentAction = Union[ToolAction, PluginAction, NullAction]
PlanableAgentAction = Union[ToolAction, PluginAction]


@dataclass
class AgentPlan(object):
    """A plan that contains a list of actions."""

    actions: Sequence[PlanableAgentAction]


@dataclass
class LLMResponse(object):
    """A response from an LLM."""

    message: AIMessage


@dataclass
class ToolResponse(object):
    """A response from a tool."""

    json: str
    input_files: Sequence[File]
    output_files: Sequence[File]


_IT = TypeVar("_IT", bound=Dict)
_RT = TypeVar("_RT")


@dataclass
class AgentStep(Generic[_IT, _RT]):
    """A step taken by an agent."""

    info: _IT
    result: _RT


@dataclass
class AgentStepWithFiles(AgentStep[_IT, _RT]):
    """A step taken by an agent involving file input and output."""

    input_files: Sequence[File]
    output_files: Sequence[File]

    @property
    def files(self) -> List[File]:
        return [*self.input_files, *self.output_files]


@dataclass
class ToolStep(AgentStepWithFiles[ToolInfo, Any]):
    """A step taken by an agent that calls a tool."""


@dataclass
class PluginStep(AgentStepWithFiles[PluginInfo, str]):
    """A step taken by an agent that calls a plugin."""


class _NullInfo(Dict):
    pass


class _NullResult(object):
    pass


@dataclass
class NoActionStep(AgentStep[_NullInfo, _NullResult]):
    """A step taken by an agent that performs no action and gives no result."""


NO_ACTION_STEP = NoActionStep(info=_NullInfo(), result=_NullResult())


@dataclass
class AgentResponse(object):
    """The final response from an agent."""

    text: str
    chat_history: Sequence[Message]
    steps: Sequence[AgentStep]
    status: Union[Literal["FINISHED"], Literal["STOPPED"]]

    @functools.cached_property  # lazy and prevent extra fime from multiple calls
    def annotations(self) -> Dict[str, List]:
        annotations = self._get_annotations()

        return annotations

    def _get_annotations(self) -> Dict[str, List]:
        # 1. split the text into parts and add file id to each part
        file_ids = protocol.extract_file_ids(self.text)

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
        annotations: Dict = {"content_parts": []}

        for file_id in split_text_list:
            if file_id in file_ids:
                for step in self.steps:
                    if isinstance(step, AgentStepWithFiles):
                        for file in step.files:
                            if file_id == file.id:
                                file_meta = file.to_dict()
                                annotations["content_parts"].append(file_meta)
            else:
                annotations["content_parts"].append({"text": file_id})

        return annotations
