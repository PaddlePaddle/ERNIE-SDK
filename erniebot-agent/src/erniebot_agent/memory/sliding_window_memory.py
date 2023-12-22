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
from erniebot_agent.memory.messages import Message


class SlidingWindowMemory(Memory):
    """
    This class controls max number of rounds of message using sliding window tactic.
    Each round contains a piece of human message and a piece of AI message.

    Args:
        max_round(int): Max number of rounds. 
        retained_round(int): The first number of rounds of memory will be preserverd. Default to 0.

    """

    def __init__(self, max_round: int, retained_round: int = 0) -> None:
        """This class controls max number of messages.

        Args:
            max_round(int): Max number of rounds(round: human message and AI message).
            retained_round(int): The number remaining_memory rounds of memory to be retained. Default to 0.

        Raises:
            ValueError: If max_round is not positive integer.
        """

        super().__init__()
        self.max_round = max_round
        self.retained_round = retained_round

        if max_round <= 0:
            raise ValueError(f"max_round should be positive integer, but got {max_round}")

    def add_message(self, message: Message) -> None:
        """Add a message to memory."""
        super().add_message(message=message)
        self.prune_message()

    def prune_message(self) -> None:
        """Prune memory to max_round if necessary."""
        while len(self.get_messages()) > self.max_round * 2:
            self.msg_manager.pop_message(self.retained_round * 2)
            if len(self.get_messages()) % 2 == 0:
                self.msg_manager.pop_message()
