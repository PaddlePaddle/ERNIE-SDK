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
from erniebot_agent.memory import AIMessage, Message


class LimitTokensMemory(Memory):
    """This class controls max tokens less than max_token_limit.
    If tokens >= max_token_limit, pop message from memory.
    """

    def __init__(self, max_token_limit=3000):
        super().__init__()
        self.max_token_limit = max_token_limit
        self.mem_token_count = 0

        assert (
            max_token_limit is None
        ) or max_token_limit > 0, "max_token_limit should be None or positive integer, \
                but got {max_token_limit}".format(
            max_token_limit=max_token_limit
        )

    def add_message(self, message: Message):
        super().add_message(message)
        # TODO(shiyutang): 仅在添加AIMessage时截断会导致HumanMessage传入到LLM时可能长度超限
        # 最优方案为每条message产生时确定token_count，从而在每次加入message时都进行prune_message
        if isinstance(message, AIMessage):
            self.prune_message()

    def prune_message(self):
        self.mem_token_count += self.msg_manager.messages[-1].token_count
        self.mem_token_count += self.msg_manager.messages[-2].token_count  # add human message token length
        if self.max_token_limit is not None:
            while self.mem_token_count > self.max_token_limit:
                deleted_message = self.msg_manager.pop_message()
                self.mem_token_count -= deleted_message.token_count
            else:
                # if delete all
                if len(self.get_messages()) == 0:
                    raise RuntimeError(
                        "The messsage is now empty. \
It indicates {} which takes up {} tokens and exeeded tokens limits of {} tokens.".format(
                            deleted_message, deleted_message.token_count, self.max_token_limit
                        )
                    )
