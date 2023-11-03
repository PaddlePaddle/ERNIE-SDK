import asyncio
import os
import unittest

from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.memory import WholeMemory
from erniebot_agent.message import HumanMessage

from tests.utils import MockErnieBot


class TestWholeMemory(unittest.TestCase):
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

    def test_whole_memory(self):
        async def run_whole_memory():
            messages = [
                HumanMessage(content="What is the purpose of model regularization?"),
            ]
            memory = WholeMemory()
            # memory =
            memory.add_messages(messages)
            message = await self.llm.async_chat(messages)
            memory.add_messages([message])
            memory.add_messages([HumanMessage("OK, what else?")])
            message = await self.llm.async_chat(memory.get_messages())
            self.assertTrue(message is not None)

        asyncio.run(run_whole_memory())


if __name__ == "__main__":
    unittest.main()
