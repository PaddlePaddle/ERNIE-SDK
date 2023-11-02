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

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, List, Optional, Union

from erniebot_agent.message import Message


class ChatModel(ABC):
    """The base class of chat-optimized LLM."""

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def async_chat(
        self, messages: List[Message], stream: Optional[bool] = False, **kwargs: Any
    ) -> Union[Message, AsyncIterator[Message]]:
        """
        Asynchronously chat with the LLM.

        Args:
            messages(List[Message]): A list of messages.
            stream(Optional[bool]): Whether to use streaming generation. Defaults to False.
            kwargs(Any): Arbitrary keyword arguments.

        Returns:
            If stream is False, returns a single message.
            If stream is True, returns an asynchronous iterator of messages.
        """
        raise NotImplementedError
