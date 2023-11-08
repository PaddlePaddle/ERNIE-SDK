import os
import unittest

import pytest
from erniebot_agent.chat_models.erniebot import ERNIEBot
from erniebot_agent.message import FunctionMessage, HumanMessage


class TestChatModel(unittest.TestCase):
    @pytest.mark.asyncio
    async def test_chat(self):
        eb = ERNIEBot(
            model="ernie-bot-turbo", api_type="aistudio", access_token=os.getenv("AISTUDIO_ACCESS_TOKEN")
        )
        messages = [
            HumanMessage(content="我在深圳，周末可以去哪里玩？"),
        ]
        res = await eb.async_chat(messages, stream=False)
        print(res)
        self.assertTrue(False)

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

        eb = ERNIEBot(
            model="ernie-bot", api_type="aistudio", access_token=os.getenv("AISTUDIO_ACCESS_TOKEN")
        )
        messages = [
            HumanMessage(content="深圳市今天的气温是多少摄氏度？"),
        ]
        res = await eb.async_chat(messages, functions=functions)
        print(res)

        messages.append(res)
        messages.append(
            FunctionMessage(name="get_current_temperature", content='{"temperature":25,"unit":"摄氏度"}')
        )
        res = await eb.async_chat(messages, functions=functions)
        print(res)
        self.assertTrue(False)
