import asyncio
import unittest

from erniebot_agent.memory import LimitTokensMemory
from erniebot_agent.message import HumanMessage

from tests.unit_tests.utils import MockErnieBot


class Testlimit_tokenMemory(unittest.TestCase):
    def setUp(self):
        self.llm = MockErnieBot(None, None, None)

    def test_limit_token_memory(self):
        async def run_limit_token_memory():
            messages = HumanMessage(content="What is the purpose of model regularization?")

            memory = LimitTokensMemory(4000)
            memory.add_message(messages)
            message = await self.llm.async_chat(messages)
            memory.add_message(message)
            memory.add_message(HumanMessage("OK, what else?"))
            message = await self.llm.async_chat(memory.get_messages())
            self.assertTrue(message is not None)

        asyncio.run(run_limit_token_memory())

    def test_limit_token_memory_truncate_tokens(self):  # truncate through returned message
        async def run_limit_token_memory_truncate_tokens(k=3):
            # The memory
            memory = LimitTokensMemory(10)

            for _ in range(k):
                # 2 times of human message
                memory.add_message(HumanMessage(content="What is the purpose of model regularization?"))

                # AI message
                message = await self.llm.async_chat(memory.get_messages())
                memory.add_message(message)

            self.assertTrue(memory.token_length <= 100)

        asyncio.run(run_limit_token_memory_truncate_tokens())


if __name__ == "__main__":
    unittest.main()
