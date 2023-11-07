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
# limitations under the License
from typing import Dict, Union

from erniebot.response import EBResponse


class Message:
    """The base class of message."""

    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

    def __str__(self) -> str:
        return f"role:{self.role}, content: {self.content}"


class MessageWithTokenLen(Message):
    """A Message with token length."""

    def __init__(self, role: str, content: str, token_len: Union[int, None] = None):
        super().__init__(role=role, content=content)
        self.content_token_length = token_len

    def set_token_len(self, token_len: int):
        """Set the token length of message."""
        if self.content_token_length is not None:
            raise ValueError("The token length of message has been set.")
        self.content_token_length = token_len

    def get_token_len(self) -> int:
        assert self.content_token_length, "The token length of message has not been set."
        return self.content_token_length

    def to_dict(self) -> Dict[str, str]:
        attribute_dict = super().to_dict()
        attribute_dict["token_len"] = str(self.content_token_length)

        return attribute_dict

    def __str__(self) -> str:
        return f"role:{self.role}, content: {self.content}, token_len: {self.content_token_length}"


class HumanMessage(MessageWithTokenLen):
    """A Message from human."""

    def __init__(self, content: str, token_len: Union[int, None] = None):
        super().__init__(role="user", content=content, token_len=token_len)


class AIMessage(MessageWithTokenLen):
    """A Message from assistant."""

    def __init__(self, content: str, token_len_infor: Dict[str, int]):
        prompt_tokens, completion_tokens = self._parse_token_len(token_len_infor)
        super().__init__(role="assistant", content=content, token_len=completion_tokens)
        self.query_tokens_len = prompt_tokens

    def _parse_token_len(self, token_len_infor: Dict[str, int]):
        """Parse the token length information from LLM."""
        return token_len_infor["prompt_tokens"], token_len_infor["completion_tokens"]


class FunctionMessage(Message):
    """A Message from assistant for function calling."""

    def __init__(self, function_call):
        super().__init__(role="assistant", content="null")
        self.function_call = function_call

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content, "function_call": self.function_call}

    def __str__(self) -> str:
        return f"role:{self.role}, content: {self.content}, function_call: {self.function_call}"


def response_to_message(response: EBResponse) -> Message:
    """Convert the response from assistant to AIMessage or FunctionMessage."""
    if hasattr(response, "function_call"):
        return FunctionMessage(function_call=response.get_result())
    else:
        return AIMessage(content=response.get_result(), token_len_infor=response.usage)
