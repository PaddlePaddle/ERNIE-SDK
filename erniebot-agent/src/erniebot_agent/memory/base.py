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

import logging
from typing import List, Optional, Union

from erniebot_agent.memory.messages import AIMessage, Message, SystemMessage

_logger = logging.getLogger(__name__)


class MessageManager:
    """
    Messages Manager, manage the messages of a conversation.

    Attributes:
        messages (List[Message]): the messages of a conversation.
        system_message (SystemMessage): the system message of a conversation.

    Note:
        Each message manager have only one system message.
    """

    def __init__(self) -> None:
        self.messages: List[Message] = []
        self._system_message: Union[SystemMessage, None] = None

    @property
    def system_message(self) -> Optional[Message]:
        return self._system_message

    @system_message.setter
    def system_message(self, message: SystemMessage) -> None:
        if self._system_message is not None:
            _logger.warning("system message has been set, the previous one will be replaced")

        self._system_message = message

    def add_messages(self, messages: List[Message]) -> None:
        self.messages.extend(messages)

    def add_message(self, message: Message) -> None:
        if isinstance(message, SystemMessage):
            self.system_message = message
        else:
            self.messages.append(message)

    def pop_message(self, index: int = 0) -> Message:
        return self.messages.pop(index)

    def clear_messages(self) -> None:
        self.messages = []

    def update_last_message_token_count(self, token_count: int):
        if token_count == 0:
            self.messages[-1].token_count = len(self.messages[-1].content)
        else:
            self.messages[-1].token_count = token_count

    def retrieve_messages(self) -> List[Message]:
        return self.messages


class Memory:
    """
    The base class of memory

    Attributes:
        msg_manager (MessageManager): the message manager of a conversation.

    Returns:
        A memory object.
    """

    def __init__(self):
        self.msg_manager = MessageManager()

    def set_system_message(self, message: SystemMessage):
        """Set the system message of a conversation."""
        self.msg_manager.system_message = message

    def add_messages(self, messages: List[Message]):
        """Add a list of messages to memory."""
        for message in messages:
            self.add_message(message)

    def add_message(self, message: Message):
        """Add a message to memory."""
        if isinstance(message, AIMessage):
            self.msg_manager.update_last_message_token_count(message.query_tokens_count)
        self.msg_manager.add_message(message)

    def get_messages(self) -> List[Message]:
        """Get all the messages in memory."""
        if self.msg_manager.system_message is not None:
            return [self.msg_manager.system_message] + self.msg_manager.retrieve_messages()
        else:
            return self.msg_manager.retrieve_messages()

    def clear_chat_history(self):
        """Reset the memory."""
        self.msg_manager.clear_messages()
