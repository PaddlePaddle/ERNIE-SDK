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

from typing import List

from erniebot_agent.message import AIMessage, MessageWithTokenLen


class MessageManager:
    """Manage messages"""

    def __init__(self):
        self.messages: List[MessageWithTokenLen] = []

    def add_messages(self, messages: List[MessageWithTokenLen]):
        self.messages.extend(messages)

    def add_message(self, message: MessageWithTokenLen):
        self.messages.append(message)

    def pop_message(self):
        return self.messages.pop(0)

    def clear_messages(self) -> None:
        self.messages = []

    def edit_last_message_token_length(self, token_len: int):
        self.messages[-1].set_token_len(token_len)

    def retrieve_messages(self):
        return self.messages


class Memory:
    """The base class of memory"""

    def __init__(self):
        self.msg_manager = MessageManager()

    def add_messages(self, messages: List[MessageWithTokenLen]):
        self.msg_manager.add_messages(messages)

    def add_message(self, message: MessageWithTokenLen):
        if isinstance(message, AIMessage):
            self.msg_manager.edit_last_message_token_length(message.query_tokens_len)
        self.msg_manager.add_message(message)

    def get_messages(self) -> List[MessageWithTokenLen]:
        return self.msg_manager.retrieve_messages()

    def clear_chat_history(self):
        self.msg_manager.clear_messages()


class WholeMemory(Memory):
    """The memory include all the messages"""

    def __init__(self):
        super().__init__()
