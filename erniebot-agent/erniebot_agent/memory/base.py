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

from typing import Dict, List, Optional, Union

from erniebot_agent.messages import AIMessage, HumanMessage, Message


class MessageManager:
    """Manage messages"""

    def __init__(self) -> None:
        self.messages: List[Message] = []

    def add_messages(self, messages: List[Message]) -> None:
        self.messages.extend(messages)

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    def pop_message(self) -> Message:
        return self.messages.pop(0)

    def clear_messages(self) -> None:
        self.messages = []

    def update_last_message_token_count(self, token_count: int):
        self.messages[-1].token_count = token_count

    def get_messages(self) -> List[Message]:
        return self.messages


user_AK_relation = {"AK-123": "user-123", "AK-124": "user-124"}
user_session_id_relation: Dict[str, List] = {"user-123": [], "user-124": ["session-124", "session-125"]}

session_messages = {
    "session-124": [HumanMessage(content="你好"), AIMessage(content="你好124", function_call=None)],
    "session-125": [HumanMessage(content="你好"), AIMessage(content="你好125", function_call=None)],
}


class RemoteMemory:
    """
    远程memory的实现类, 用于管理一个user 在一个session中的messages。
    """

    def __init__(self, user_id, session_id):
        self.session_id: int = session_id
        self.messages: list[Message] = session_messages[session_id]

    def add_message(self, message):
        "make changes to the session's memory"
        session_messages[self.session_id].append(message)

    def pop_message(self):
        """pop the message from the start"""
        session_messages[self.session_id].pop(0)

    def clear_memory(self):
        session_messages[self.session_id] = []

    def get_messages(self):
        if self.session_id not in session_messages.keys():
            raise KeyError(f"session_id {self.session_id} not found")
        return session_messages[self.session_id]

    def search_memory(self, session_id, payload):  # TODO: refer zep
        pass

    # TODO: 关闭之后同步message的变化到数据库


class MessageStorageServer:  # 绑定user
    """
    MessageStorageServer 用于管理一个user在多个session中的message切换。

    Args:
        request_url (str): 请求地址
        AK (str): 用户ID
        session_id (str, optional): 用户选择的session对应的session id. Defaults to None.
    """

    def __init__(self, request_url: str, AK: str, session_id: Optional[str] = None):
        self.request_url = request_url
        self.AK = AK
        self.user_id = user_AK_relation[AK]
        self.sessions: List = user_session_id_relation[self.user_id]
        if len(self.sessions) == 0:
            self.create_session()

        self.session_id = session_id if session_id else self.sessions[-1]  # TODO: session选择
        self.memory = RemoteMemory(self.user_id, self.session_id)

    def get_messages(self):
        return self.memory.get_messages()

    def create_session(
        self,
    ):
        """create a new session for user and return the session id"""
        import uuid

        session_id = uuid.uuid4().hex  # A new session identifier
        self.sessions.append(session_id)
        user_session_id_relation[self.user_id] = [session_id]
        global session_messages
        session_messages[session_id] = []
        # 同时在数据库中创建相应空间
        return session_id


class PersistentMessageManager:
    """
    PersistentMessageManager 用于本地的持久化、隔离化message管理。
    """

    def __init__(self, url: str, AK: str, session_id: Optional[str] = None):
        self.client = MessageStorageServer(
            request_url=url, AK=AK, session_id=session_id
        )  # client 内确定了session_id
        self.session_id = self.client.session_id  # 统一内外的session_id
        self.messages = self.get_messages()

    def add_message(self, message: Message):
        self.client.memory.add_message(message=[message])
        self.messages.append(message)

    def clear_messages(self):
        self.messages = []
        self.client.memory.clear_memory()

    def pop_message(self):  # TODO: choose from pop_message and cherry_pick_message
        delete_message = self.client.memory.pop_message()
        return delete_message

    def get_messages(
        self,
    ) -> List[Message]:  # system,AI,user,contains summary if necessary
        memory = self.client.memory.get_messages()
        return memory

    # def cherry_pick_message(self, query): # TODO: 不使用pop，而是利用存储后端的索引功能找到相关message，但不保证限制长度
    #     from zep_python import MemorySearchPayload

    #     payload: MemorySearchPayload = MemorySearchPayload(text=query)

    #     return self.client.memory.search_memory(self.session_id, payload)

    def update_last_message_token_count(self, token_count: int):
        self.client.memory.get_messages()[-1].token_count = token_count


class Memory:
    """The base class of memory"""

    def __init__(self, message_manager: Union[PersistentMessageManager, MessageManager] = MessageManager()):
        self.msg_manager = message_manager

    def add_messages(self, messages: List[Message]):
        for message in messages:
            self.add_message(message)

    def add_message(self, message: Message):
        if isinstance(message, AIMessage):
            self.msg_manager.update_last_message_token_count(message.query_tokens_count)
        self.msg_manager.add_message(message)

    def get_messages(self) -> List[Message]:
        return self.msg_manager.get_messages()

    def clear_chat_history(self):
        self.msg_manager.clear_messages()


class WholeMemory(Memory):
    """The memory include all the messages"""

    def __init__(self):
        super().__init__()
