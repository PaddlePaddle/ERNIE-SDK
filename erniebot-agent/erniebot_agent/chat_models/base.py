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

from abc import abstractmethod, ABCMeta
from typing import Any, List, Union, AsyncIterator, overload, Literal

from erniebot_agent.messages import Message


class ChatModel(metaclass=ABCMeta):
    """The base class of chat-optimized LLM."""

    model: str

    def __init__(self, model: str) -> None:
        self.model = model

    @overload
    async def run(self,
                  messages: List[Message],
                  *,
                  stream: Literal[False]=...,
                  **kwargs: Any) -> Message:
        ...

    @overload
    async def run(self,
                  messages: List[Message],
                  *,
                  stream: Literal[True],
                  **kwargs: Any) -> AsyncIterator[Message]:
        ...

    @overload
    async def run(self, messages: List[Message], *, stream: bool,
                  **kwargs: Any) -> Union[Message, AsyncIterator[Message]]:
        ...

    @abstractmethod
    async def run(self,
                  messages: List[Message],
                  *,
                  stream: bool=False,
                  **kwargs: Any) -> Union[Message, AsyncIterator[Message]]:
        """
        Asynchronously chat with the LLM.

        Args:
            messages (List[Message]): A list of messages.
            stream (bool): Whether to use streaming generation. Defaults to False.
            kwargs (Any): Arbitrary keyword arguments.

        Returns:
            If stream is False, returns a single message.
            If stream is True, returns an asynchronous iterator of messages.
        """
        raise NotImplementedError
