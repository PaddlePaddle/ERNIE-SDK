import asyncio
import unittest

from erniebot_agent.memory import LimitTokenMemory
from erniebot_agent.message import HumanMessage
from erniebot_agent.utils import MockErnieBot


class Testlimit_tokenMemory(unittest.TestCase):
    def setUp(self):
        self.llm = MockErnieBot(None, None, None)

    def test_limit_token_memory(self):
        async def run_limit_token_memory():
            messages = [
                HumanMessage(content="What is the purpose of model regularization?"),
            ]
            memory = LimitTokenMemory(4000)
            # memory =
            memory.add_messages(messages)
            message = await self.llm.async_chat(messages)
            memory.add_messages([message])
            memory.add_messages([HumanMessage("OK, what else?")])
            message = await self.llm.async_chat(memory.get_messages())
            self.assertTrue(message is not None)

        asyncio.run(run_limit_token_memory())

    def test_limit_token_memory_exceed_tokens(self):
        messages = [
            HumanMessage(content="What is the purpose of model regularization?"),
        ]
        memory = LimitTokenMemory(10)
        with self.assertRaises(RuntimeError):
            memory.add_messages(messages)

    def test_limit_token_memory_truncate_tokens(self):  # truncate through returned message
        async def run_limit_token_memory_truncate_tokens(k=3):
            # The memory
            memory = LimitTokenMemory(100)

            for _ in range(k):
                # 2 times of human message
                memory.add_messages([HumanMessage(content="What is the purpose of model regularization?")])

                # AI message
                message = await self.llm.async_chat(memory.get_messages())
                memory.add_messages([message])

            self.assertTrue(memory.token_length <= 100)

        asyncio.run(run_limit_token_memory_truncate_tokens())


if __name__ == "__main__":
    unittest.main()
