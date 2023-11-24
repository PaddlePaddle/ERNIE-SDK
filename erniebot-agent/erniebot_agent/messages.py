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
from typing import Dict, Optional, TypedDict


class Message:
    """The base class of a message."""

    def __init__(self, role: str, content: str, token_count: Optional[int] = None):
        self.role = role
        self.content = content
        self._token_count = token_count
        self._param_names = ["role", "content"]

    @property
    def token_count(self):
        """Get the number of tokens of the message."""
        assert self._token_count, "The token length of message has not been set before get the token length."
        return self._token_count

    @token_count.setter
    def token_count(self, token_count: int):
        """Set the number of tokens of the message."""
        if self._token_count is not None:
            raise ValueError("The token length of message has been set.")
        self._token_count = token_count

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
        else:
            res += f"token_count: {self._token_count}"
        return f"<{res[:-2]}>"

    def __repr__(self):
        return self.__str__()


class SystemMessage(Message):
    """A message from human to set system information."""

    def __init__(self, content: str):
        super().__init__(role="system", content=content)


class HumanMessage(Message):
    """The message from human."""

    def __init__(self, content: str):
        super().__init__(role="user", content=content)


class FunctionCall(TypedDict):
    name: str
    thoughts: str
    arguments: str


class TokenUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int


class AIMessage(Message):
    """A Message from assistant."""

    def __init__(
        self,
        content: str,
        function_call: Optional[FunctionCall],
        token_usage: Optional[TokenUsage] = None,
    ):
        if token_usage is None:
            prompt_tokens = 0
            completion_tokens = len(content)
            Warning(
                "The token usage is not set in AIMessage,\
                     the token counts of AIMessage and HumanMessage are not correct."
            )
        else:
            prompt_tokens, completion_tokens = self._parse_token_count(token_usage)
        super().__init__(role="assistant", content=content, token_count=completion_tokens)
        self.function_call = function_call
        self.query_tokens_count = prompt_tokens
        self._param_names = ["role", "content", "function_call"]

    def _parse_token_count(self, token_usage: TokenUsage):
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
    token_usage: Optional[TokenUsage]
