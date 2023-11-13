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
from dataclasses import dataclass
from typing import Dict, Optional, TypedDict, Union


class Message:
    """The base class of a message."""

    def __init__(self, role: str, content: str):
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
    """A message from human to set system information."""

    def __init__(self, content: str):
        super().__init__(role="system", content=content)


class MessageWithTokenLen(Message):
    """A Message with token length."""

    def __init__(self, role: str, content: str, token_len: Union[int, None] = None):
        super().__init__(role=role, content=content)
        self.token_len = token_len
        self._param_names = ["role", "content", "token_len"]

    def set_token_len(self, token_len: int):
        """Set the number of tokens of the message."""
        if self.token_len is not None:
            raise ValueError("The token length of message has been set.")
        self.token_len = token_len

    def get_token_len(self) -> int:
        """Get the number of tokens of the message."""
        assert self.token_len, "The token length of message has not been set before get the token length."
        return self.token_len

    def __str__(self) -> str:
        return f"role:{self.role}, content: {self.content}, token_len: {self.token_len}"


class HumanMessage(MessageWithTokenLen):
    """The message from human."""

    def __init__(self, content: str, token_len: Union[int, None] = None):
        super().__init__(role="user", content=content, token_len=token_len)


class FunctionCall(TypedDict):
    name: str
    thoughts: str
    arguments: str


class AIMessage(MessageWithTokenLen):
    """A Message from assistant."""

    def __init__(
        self,
        content: str,
        function_call: Optional[FunctionCall],
        token_usage: Dict[str, int],
    ):
        prompt_tokens, completion_tokens = self._parse_token_len(token_usage)
        super().__init__(role="assistant", content=content, token_len=completion_tokens)
        self.function_call = function_call
        self.query_tokens_len = prompt_tokens
        self._param_names = ["role", "content", "function_call"]

    def _parse_token_len(self, token_usage: Dict[str, int]):
        """Parse the token length information from LLM."""
        return token_usage["prompt_tokens"], token_usage["completion_tokens"]


class FunctionMessage(Message):
    """A message from human that contains the result of a function call."""

    def __init__(self, name: str, content: str):
        super().__init__(role="function", content=content)
        self.name = name
        self._param_names = ["role", "name", "content"]


@dataclass
class AIMessageChunk(object):
    content: str
    function_call: Optional[FunctionCall]
    token_usage: Dict[str, int]
