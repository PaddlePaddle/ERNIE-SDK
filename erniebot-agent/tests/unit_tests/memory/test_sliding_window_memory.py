import unittest

import pytest
from erniebot_agent.memory import SlidingWindowMemory
from erniebot_agent.messages import HumanMessage

from tests.unit_tests.testing_utils.mocks.mock_chat_models import FakeSimpleChatModel


class TestSlidingWindowMemory(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.llm = FakeSimpleChatModel()

    @pytest.mark.parametrize("k", [1, 2, 4, 5, 10])
    @pytest.mark.asyncio
    async def test_sliding_window_memory(self, k=2):  # asyn pytest
        # The memory
        memory = SlidingWindowMemory(k)

        for _ in range(k):
            # 2 times of human message
            memory.add_message(HumanMessage(content="What is the purpose of model regularization?"))

            # AI message
            message = await self.llm.async_chat(memory.get_messages())
            memory.add_message(message)

        self.assertTrue(len(memory.get_messages()) <= 2 * k)


if __name__ == "__main__":
    unittest.main()
