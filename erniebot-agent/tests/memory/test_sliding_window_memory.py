import asyncio
import os
import unittest

from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.memory import SlidingWindowMemory
from erniebot_agent.message import HumanMessage

from tests.utils import MockErnieBot


class TestSlidingWindowMemory(unittest.TestCase):
    def setUp(self, mockllm=True):
        if mockllm:
            self.llm = MockErnieBot(None, None, None)
        else:
            access_token = os.getenv("access_token")
            self.llm = ERNIEBot(
                model="ernie-bot",
                api_type="aistudio",
                access_token=access_token,
            )

    def test_sliding_window_memory(self):  # asyn pytest
        async def run_sliding_window_memory(k=3):
            # The memory
            memory = SlidingWindowMemory(k)

            for _ in range(k):
                # 2 times of human message
                memory.add_messages([HumanMessage(content="What is the purpose of model regularization?")])

                # AI message
                message = await self.llm.async_chat(memory.get_messages())
                memory.add_messages([message])

            self.assertTrue(len(memory.get_messages()) <= k)

        asyncio.run(run_sliding_window_memory())


if __name__ == "__main__":
    unittest.main()
