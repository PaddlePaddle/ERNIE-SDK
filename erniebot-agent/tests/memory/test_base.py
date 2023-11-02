import os
import asyncio
import unittest

from erniebot_agent.memory import WholeMemory, BufferWindowMemory
from erniebot_agent.message import HumanMessage
from erniebot_agent.chat_models import ERNIEBot

class TestMemory(unittest.TestCase):
    def setUp(self) -> None:
        access_token=os.getenv('access_token')
        self.llm = ERNIEBot(
                model='ernie-bot',
                api_type='aistudio',
                access_token=access_token,
            )
        return super().setUp()

    def test_whole_memory(self):
        async def run_whole_memory():
            messages = [
                HumanMessage(content="What is the purpose of model regularization?"),
            ] 
            memory = WholeMemory(4000) 
            # memory = 
            memory.add_message(messages) 
            message = await self.llm.async_chat(messages) 
            memory.add_message([message])
            memory.add_message([HumanMessage('OK, what else?')])
            message = await self.llm.async_chat(memory.get_messages())
            self.assertTrue(message is not None)
            
        asyncio.run(run_whole_memory())
    
    def test_whole_memory_exceed_tokens(self):
        messages = [
            HumanMessage(content="What is the purpose of model regularization?"),
        ] 
        memory = WholeMemory(10) 
        with self.assertRaises(RuntimeError):
            memory.add_message(messages) 


    def test_buffer_window_memory(self):

        async def run_buffer_window_memory(k=3):
            # The memory
            memory = BufferWindowMemory(k) 

            # human message
            memory.add_message([HumanMessage(content="What is the purpose of model regularization?")])

            # AI message
            message = await self.llm.async_chat(memory.get_messages())
            memory.add_message([message])
            
            # human message
            for _ in range(2):
                # 10 times of human message
                memory.add_message([HumanMessage('OK, what else?')]) 

                message = await self.llm.async_chat(memory.get_messages())
                memory.add_message([message]) 
                    
            self.assertTrue(len(memory.get_messages())<=k)
        
        asyncio.run(run_buffer_window_memory())


if __name__ == '__main__':
    unittest.main()