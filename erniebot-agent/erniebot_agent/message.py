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
from typing import Dict, Optional, Union

from erniebot.response import EBResponse


class Message:
    """The base class of message."""

    def __init__(self, role: str, content: Optional[str]):
        self.role = role
        self.content = content
        self._param_names = ["role", "content"]

    def to_dict(self) -> Dict[str, str]:
        res = {}
        for name in self._param_names:
            res[name] = getattr(self, name)
        return res

    def __str__(self) -> str:
        res = ""
        for name in self._param_names:
            value = getattr(self, name)
            if value is not None and value != "":
                res += f"{name}: {value}, "
        return res[:-2]


class SystemMessage(Message):
    """The message from human to set system information."""

    def __init__(self, content: str):
        super().__init__(role="system", content=content)


class MessageWithTokenLen(Message):
    """A Message with token length."""

    def __init__(self, role: str, content: Optional[str], token_len: Union[int, None] = None):
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
    """The message from human."""

    def __init__(self, content: str, token_len: Union[int, None] = None):
        super().__init__(role="user", content=content, token_len=token_len)


class AIMessage(MessageWithTokenLen):
    """A Message from assistant."""

    def __init__(
        self,
        content: Optional[str],
        function_call: Optional[Dict[str, str]],
        token_len_infor: Dict[str, int],
    ):
        prompt_tokens, completion_tokens = self._parse_token_len(token_len_infor)

        super().__init__(role="assistant", content=content, token_len=completion_tokens)

        self.function_call = function_call
        self.query_tokens_len = prompt_tokens
        self._param_names = ["role", "content", "function_call"]

    def _parse_token_len(self, token_len_infor: Dict[str, int]):
        """Parse the token length information from LLM."""
        return token_len_infor["prompt_tokens"], token_len_infor["completion_tokens"]

    @classmethod
    def from_response(cls, response: EBResponse):
        if hasattr(response, "function_call"):
            return cls(content=None, function_call=response.function_call, token_len_infor=response.usage)
        else:
            return cls(content=response.result, function_call=None, token_len_infor=response.usage)


class FunctionMessage(Message):
    """The message from human to set the result of function call."""

    def __init__(self, name: str, content: str):
        super().__init__(role="function", content=content)
        self.name = name
        self._param_names = ["role", "name", "content"]
