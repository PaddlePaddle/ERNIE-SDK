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

from erniebot_agent.memory import Memory, MessageManager
from erniebot_agent.messages import Message


class SlidingWindowMemory(Memory):
    """This class controls max number of messages."""

    def __init__(self, max_num_message: int, message_manager=MessageManager()):
        super().__init__(message_manager)
        self.max_num_message = max_num_message

        assert (isinstance(max_num_message, int)) and (
            max_num_message > 0
        ), "max_num_message should be positive integer, but got {max_token_limit}".format(
            max_token_limit=max_num_message
        )

    def add_message(self, message: Message):
        super().add_message(message=message)
        self.prune_message()

    def prune_message(self):
        while len(self.get_messages()) > self.max_num_message:
            self.msg_manager.pop_message()
            # `messages` must have an odd number of elements.
            if len(self.get_messages()) % 2 == 0:
                self.msg_manager.pop_message()
