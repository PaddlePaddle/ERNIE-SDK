import asyncio
import unittest

from erniebot_agent.memory import WholeMemory
from erniebot_agent.message import HumanMessage

from tests.unit_tests.utils import MockErnieBot


class TestWholeMemory(unittest.TestCase):
    def setUp(self):
        self.llm = MockErnieBot(None, None, None)

    def test_whole_memory(self):
        async def run_whole_memory():
            message = (HumanMessage(content="What is the purpose of model regularization?"),)

            memory = WholeMemory()
            memory.add_message(message)
            message = await self.llm.async_chat(message)
            memory.add_message(message)
            memory.add_message(HumanMessage("OK, what else?"))
            message = await self.llm.async_chat(memory.get_messages())
            self.assertTrue(message is not None)

        asyncio.run(run_whole_memory())


if __name__ == "__main__":
    unittest.main()
