import os
import unittest

import pytest

from erniebot_agent.chat_models.erniebot import ERNIEBot
from erniebot_agent.message import AIMessage, FunctionMessage, HumanMessage


class TestChatModel(unittest.IsolatedAsyncioTestCase):
    @pytest.mark.asyncio
    async def test_chat(self):
        eb = ERNIEBot(
            model="ernie-turbo", api_type="aistudio", access_token=os.environ["AISTUDIO_ACCESS_TOKEN"]
        )
        messages = [
            HumanMessage(content="你好！"),
        ]
        res = await eb.async_chat(messages, stream=False)
        self.assertTrue(isinstance(res, AIMessage))
        self.assertIsNotNone(res.content)

    @pytest.mark.asyncio
    async def test_function_call(self):
        functions = [
            {
                "name": "get_current_temperature",
                "description": "获取指定城市的气温",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "城市名称",
                        },
                        "unit": {
                            "type": "string",
                            "enum": [
                                "摄氏度",
                                "华氏度",
                            ],
                        },
                    },
                    "required": [
                        "location",
                        "unit",
                    ],
                },
                "responses": {
                    "type": "object",
                    "properties": {
                        "temperature": {
                            "type": "integer",
                            "description": "城市气温",
                        },
                        "unit": {
                            "type": "string",
                            "enum": [
                                "摄氏度",
                                "华氏度",
                            ],
                        },
                    },
                },
            }
        ]
        # NOTE：use ernie-3.5 here since ernie-turbo doesn't support function call
        eb = ERNIEBot(
            model="ernie-3.5", api_type="aistudio", access_token=os.environ["AISTUDIO_ACCESS_TOKEN"]
        )
        messages = [
            HumanMessage(content="深圳市今天的气温是多少摄氏度？"),
        ]
        res = await eb.async_chat(messages, functions=functions)
        self.assertTrue(isinstance(res, AIMessage))
        self.assertIsNone(res.content)
        self.assertIsNotNone(res.function_call)
        self.assertEqual(res.function_call["name"], "get_current_temperature")

        messages.append(res)
        messages.append(
            FunctionMessage(name="get_current_temperature", content='{"temperature":25,"unit":"摄氏度"}')
        )
        res = await eb.async_chat(messages, functions=functions)
        self.assertTrue(isinstance(res, AIMessage))
        self.assertIsNotNone(res.content)
