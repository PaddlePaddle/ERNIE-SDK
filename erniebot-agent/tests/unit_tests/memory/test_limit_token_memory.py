import unittest

import pytest
from tests.unit_tests.testing_utils.mocks.mock_chat_models import FakeSimpleChatModel

from erniebot_agent.memory import HumanMessage, LimitTokensMemory, SystemMessage


class Testlimit_tokenMemory(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.llm = FakeSimpleChatModel()

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
        memory = LimitTokensMemory(20)

        for _ in range(k):
            # 2 times of human message
            memory.add_message(HumanMessage(content="What is the purpose of model regularization?"))

            # AI message
            message = await self.llm.async_chat(memory.get_messages())
            memory.add_message(message)

        self.assertTrue(memory.mem_token_count <= 20)

    @pytest.mark.asyncio
    async def test_limit_token_memory_truncate_tokens_system_message(
        self, k=10
    ):  # truncate through returned message
        # The memory
        memory = LimitTokensMemory(100)

        # Keypoint 1: system message 的存取
        memory.system_message = SystemMessage("你是一个善于回答图像相关问题的agent。")
        self.assertTrue(memory.system_message.content == "你是一个善于回答图像相关问题的agent。")

        for _ in range(k):
            # 2 times of human message
            memory.add_message(HumanMessage(content="这个图像中的内容是什么？"))

            # AI message
            message = await self.llm.async_chat(memory.get_messages())
            memory.add_message(message)

        # Keypoint 2:没有传入token_count 的fallback情况，此时也能正确裁剪信息
        self.assertTrue(memory.mem_token_count <= 100)
        self.assertTrue(len(memory.get_messages()) < 2 * k)


if __name__ == "__main__":
    unittest.main()
