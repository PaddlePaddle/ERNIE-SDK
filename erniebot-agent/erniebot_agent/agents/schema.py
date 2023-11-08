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
from typing import List, Union

from erniebot_agent.messages import Message
from typing_extensions import Literal


@dataclass
class AgentAction(object):
    """An action for an agent to execute."""

    tool_name: str
    tool_args: str


@dataclass
class AgentResponse(object):
    """The final response of an agent."""

    content: str
    chat_history: List[Message]
    actions: List[AgentAction]
    status: Union[Literal["FINISHED"], Literal["STOPPED"]]


@dataclass
class AgentPlan(object):
    """A plan that contains a list of actions."""

    actions: List[AgentAction]
