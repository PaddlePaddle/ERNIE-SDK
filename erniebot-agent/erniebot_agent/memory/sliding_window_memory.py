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

from erniebot_agent.memory import Memory
from erniebot_agent.messages import Message


class SlidingWindowMemory(Memory):
    """This class controls max number of messages."""

    def __init__(self, max_round: int, remaining_memory: int = 0) -> None:
        """This class controls max number of messages.

        Args:
        max_num_message: Max number of rounds(round: human message and AI message).
        remaining_memory: The first K rounds of memory to be retained. Default to 0.
        """

        super().__init__()
        self.max_round = max_round
        self.remaining_memory = remaining_memory

        assert (isinstance(max_round, int)) and (
            max_round > 0
        ), "max_num_message should be positive integer, but got {max_token_limit}".format(
            max_token_limit=max_round
        )

    def add_message(self, message: Message) -> None:
        super().add_message(message=message)
        self.prune_message()

    def prune_message(self) -> None:
        while len(self.get_messages()) > self.max_round * 2:
            self.msg_manager.pop_message(self.remaining_memory * 2)
            # `messages` must have an odd number of elements.
            if len(self.get_messages()) % 2 == 0:
                self.msg_manager.pop_message()
