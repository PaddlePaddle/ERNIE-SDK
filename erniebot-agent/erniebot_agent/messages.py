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

from typing import Dict, List, Optional, TypedDict

from erniebot_agent.file_io.base import File

import erniebot.utils.token_helper as token_helper


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
        if self._token_count is None:
            raise AttributeError("The token count of the message has not been set.")
        return self._token_count

    @token_count.setter
    def token_count(self, token_count: int):
        """Set the number of tokens of the message."""
        if self._token_count is not None:
            raise AttributeError("The token count of the message can only be set once.")
        self._token_count = token_count

    def to_dict(self) -> Dict[str, str]:
        res = {}
        for name in self._param_names:
            res[name] = getattr(self, name)
        return res

    def __str__(self) -> str:
        return f"<{self._get_attrs_str()}>"

    def __repr__(self):
        return f"<{self.__class__.__name__} {self._get_attrs_str()}>"

    def _get_attrs_str(self) -> str:
        parts: List[str] = []
        for name in self._param_names:
            value = getattr(self, name)
            if value is not None and value != "":
                parts.append(f"{name}: {repr(value)}")
        if self._token_count is not None:
            parts.append(f"token_count: {self._token_count}")
        return ", ".join(parts)


class SystemMessage(Message):
    """A message from a human to set system information."""

    def __init__(self, content: str):
        super().__init__(role="system", content=content, token_count=len(content))


class HumanMessage(Message):
    """A message from a human."""

    def __init__(self, content: str, files: Optional[List[File]] = None):
        self.files = files
        if self.files is not None:
            prompt_parts = ["这句话中包含的文件如下："] + [
                f"<file_id>{file.id}</file_id><file>{file.filename}</file><url>{file.URL}</url>"
                for file in self.files
            ]
            prompt = "\n".join(prompt_parts)
            content = content + prompt
        super().__init__(role="user", content=content)


class FunctionCall(TypedDict):
    name: str
    thoughts: str
    arguments: str


class TokenUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int


class AIMessage(Message):
    """A message from an assistant."""

    def __init__(
        self,
        content: str,
        function_call: Optional[FunctionCall],
        token_usage: Optional[TokenUsage] = None,
    ):
        if token_usage is None:
            prompt_tokens = 0
            completion_tokens = token_helper.approx_num_tokens(content)
        else:
            prompt_tokens, completion_tokens = self._parse_token_count(token_usage)
        super().__init__(role="assistant", content=content, token_count=completion_tokens)
        self.function_call = function_call
        self.query_tokens_count = prompt_tokens
        self._param_names = ["role", "content", "function_call"]

    def _parse_token_count(self, token_usage: TokenUsage):
        """Parse the token count information from LLM."""
        return token_usage["prompt_tokens"], token_usage["completion_tokens"]


class FunctionMessage(Message):
    """A message from a human, containing the result of a function call."""

    def __init__(self, name: str, content: str):
        super().__init__(role="function", content=content)
        self.name = name
        self._param_names = ["role", "name", "content"]


class AIMessageChunk(AIMessage):
    """A message chunk from an assistant."""
