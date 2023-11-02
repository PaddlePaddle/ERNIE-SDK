# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

from typing import Dict

from erniebot.resources.chat_completion import ChatResponse


class Message:
    """The base class of message."""

    def __init__(self, role, content):
        self.role = role
        self.content = content

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

    def __str__(self) -> str:
        return f"role:{self.role}, content: {self.content}"


class HumanMessage(Message):
    """A Message from human."""

    def __init__(self, content):
        super().__init__(role="user", content=content)


class AIMessage(Message):
    """A Message from assistant."""

    def __init__(self, content):
        super().__init__(role="assistant", content=content)


class FunctionMessage(Message):
    """A Message from assistant for function calling."""

    def __init__(self, function_call):
        super().__init__(role="assistant", content="null")
        self.function_call = function_call

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content, "function_call": self.function_call}

    def __str__(self) -> str:
        return f"role:{self.role}, content: {self.content}, function_call: {self.function_call}"


def response_to_message(response: ChatResponse):
    """Convert the response from assistant to AIMessage or FunctionMessage."""
    if hasattr(response, "function_call"):
        return FunctionMessage(function_call=response.get_result())
    else:
        return AIMessage(content=response.get_result())
