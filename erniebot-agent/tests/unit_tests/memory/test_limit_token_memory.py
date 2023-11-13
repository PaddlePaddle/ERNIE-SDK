import unittest

import pytest
from erniebot_agent.memory import LimitTokensMemory
from erniebot_agent.messages import HumanMessage

from tests.unit_tests.testing_utils import MockErnieBot


class Testlimit_tokenMemory(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.llm = MockErnieBot(None, None, None)

    @pytest.mark.asyncio
    async def test_limit_token_memory(self):
        messages = HumanMessage(content="What is the purpose of model regularization?")

        memory = LimitTokensMemory(4000)
        memory.add_message(messages)
        message = await self.llm.async_chat([messages])
        memory.add_message(message)
        memory.add_message(HumanMessage("OK, what else?"))
        message = await self.llm.async_chat(memory.get_messages())
        self.assertTrue(message is not None)

    @pytest.mark.asyncio
    async def test_limit_token_memory_truncate_tokens(self, k=3):  # truncate through returned message
        # The memory
        memory = LimitTokensMemory(4)

        for _ in range(k):
            # 2 times of human message
            memory.add_message(HumanMessage(content="What is the purpose of model regularization?"))

            # AI message
            message = await self.llm.async_chat(memory.get_messages())
            memory.add_message(message)

        self.assertTrue(memory.token_length <= 4)


if __name__ == "__main__":
    unittest.main()
