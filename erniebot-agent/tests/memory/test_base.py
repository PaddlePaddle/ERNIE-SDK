import asyncio
import os
import unittest

from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.memory import BufferWindowMemory, WholeMemory
from erniebot_agent.message import HumanMessage

from tests.utils import MarkErnieBot


class TestMemory(unittest.TestCase):
    def setUp(self, markllm=True):
        if markllm:
            self.llm = MarkErnieBot(None, None, None)
        else:
            access_token = os.getenv("access_token")
            self.llm = ERNIEBot(
                model="ernie-bot",
                api_type="aistudio",
                access_token=access_token,
            )

    def test_whole_memory(self):
        async def run_whole_memory():
            messages = [
                HumanMessage(content="What is the purpose of model regularization?"),
            ]
            memory = WholeMemory(4000)
            # memory =
            memory.add_messages(messages)
            message = await self.llm.async_chat(messages)
            memory.add_messages([message])
            memory.add_messages([HumanMessage("OK, what else?")])
            message = await self.llm.async_chat(memory.get_messages())
            self.assertTrue(message is not None)

        asyncio.run(run_whole_memory())

    def test_whole_memory_exceed_tokens(self):
        messages = [
            HumanMessage(content="What is the purpose of model regularization?"),
        ]
        memory = WholeMemory(10)
        with self.assertRaises(RuntimeError):
            memory.add_messages(messages)

    def test_whole_memory_truncate_tokens(self):  # truncate through returned message
        async def run_whole_memory_truncate_tokens(k=3):
            # The memory
            memory = WholeMemory(100)

            for _ in range(k):
                # 2 times of human message
                memory.add_messages([HumanMessage(content="What is the purpose of model regularization?")])

                # AI message
                message = await self.llm.async_chat(memory.get_messages())
                memory.add_messages([message])

            self.assertTrue(memory.token_length <= 100)

        asyncio.run(run_whole_memory_truncate_tokens())

    def test_buffer_window_memory(self):  # asyn pytest
        async def run_buffer_window_memory(k=3):
            # The memory
            memory = BufferWindowMemory(k)

            for _ in range(k):
                # 2 times of human message
                memory.add_messages([HumanMessage(content="What is the purpose of model regularization?")])

                # AI message
                message = await self.llm.async_chat(memory.get_messages())
                memory.add_messages([message])

            self.assertTrue(len(memory.get_messages()) <= k)

        asyncio.run(run_buffer_window_memory())


if __name__ == "__main__":
    unittest.main()
