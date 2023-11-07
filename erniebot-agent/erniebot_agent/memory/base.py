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

from erniebot_agent.messages import Message


class MessageManager:
    """Manage messages"""

    def __init__(self) -> None:
        self.messages: List[Message] = []

    def add_messages(self, messages: List[Message]) -> None:
        for message in messages:
            self.add_message(message)

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    def pop_message(self) -> Message:
        return self.messages.pop(0)

    def clear_messages(self) -> None:
        self.messages = []

    def retrieve_messages(self) -> List[Message]:
        return self.messages


class Memory:
    def __init__(self):
        self.msg_manager = MessageManager()

    def add_messages(self, messages: List[Message]):
        self.msg_manager.add_messages(messages)

    def add_message(self, message: Message):
        self.msg_manager.add_message(message)

    def get_messages(self) -> List[Message]:
        return self.msg_manager.retrieve_messages()

    def clear_chat_history(self):
        self.msg_manager.clear_messages()
