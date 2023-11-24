import unittest

import pytest
from erniebot_agent.memory import WholeMemory
from erniebot_agent.messages import AIMessage, HumanMessage

from tests.unit_tests.testing_utils import MockErnieBot


class TestWholeMemory(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.llm = MockErnieBot(None, None, None)

    @pytest.mark.asyncio
    async def test_whole_memory(self):
        message = HumanMessage(content="What is the purpose of model regularization?")

        memory = WholeMemory()
        memory.add_message(message)
        message = await self.llm.async_chat([message])
        memory.add_message(message)
        memory.add_message(HumanMessage("OK, what else?"))
        message = await self.llm.async_chat(memory.get_messages())
        self.assertTrue(message is not None)

    def test_list_message_print_msg(self):
        messages = [HumanMessage("A"), AIMessage("B", function_call=None)]
        self.assertEqual(
            str(messages),
            (
                "[<HumanMessage role: 'user', content: 'A'>"
                ", <AIMessage role: 'assistant', content: 'B', token_count: 1>]"
            ),
        )


if __name__ == "__main__":
    unittest.main()
