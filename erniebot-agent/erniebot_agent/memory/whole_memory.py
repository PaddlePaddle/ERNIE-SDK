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
from erniebot_agent.message import Message
from typing import List

class WholeMemory(Memory):
    """This class controls max tokens less than max_token_limit.
        If tokens >= max_token_limit, pop message from memory.
    """
    def __init__(self, max_token_limit=None):
        super().__init__()
        self.max_token_limit = max_token_limit
        self.token_length = 0

        assert (max_token_limit is None) or max_token_limit > 0, "max_token_limit should be None or positive integer, but got {max_token_limit}".format(max_token_limit=max_token_limit)    
    
    def add_message(self, message: List[Message]):
        super().add_message()
        self.prune_message(message)
    
    def prune_message(self, message):
        for m in message:
            self.token_length += len(m.content)
        if self.max_token_limit is not None:
            while self.token_length > self.max_token_limit:
                deleted_message = self.chat_history.delete_message()
                self.token_length -= len(deleted_message.content)
            # if delete all 
            if len(self.get_messages()) == 0:
                raise RuntimeError('The messsage is now empty. It indicates {} exeeded {} tokens.'.format(deleted_message, self.max_token_limit))