import unittest

import pytest

from erniebot_agent.chat_models.erniebot import ERNIEBot
from erniebot_agent.memory import AIMessage, FunctionMessage, HumanMessage


class TestChatModel(unittest.IsolatedAsyncioTestCase):
    @pytest.mark.asyncio
    async def test_chat(self):
        eb = ERNIEBot(model="ernie-turbo", api_type="aistudio")
        messages = [
            HumanMessage(content="你好！"),
        ]
        res = await eb.chat(messages, stream=False)
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
        eb = ERNIEBot(model="ernie-3.5", api_type="aistudio")
        messages = [
            HumanMessage(content="深圳市今天的气温是多少摄氏度？"),
        ]
        res = await eb.chat(messages, functions=functions)
        self.assertTrue(isinstance(res, AIMessage))

        content = res.content or None
        self.assertIsNone(content)
        self.assertIsNotNone(res.function_call)
        self.assertEqual(res.function_call["name"], "get_current_temperature")

        messages.append(res)
        messages.append(
            FunctionMessage(name="get_current_temperature", content='{"temperature":25,"unit":"摄氏度"}')
        )
        res = await eb.chat(messages, functions=functions)
        self.assertTrue(isinstance(res, AIMessage))

        content = res.content or None
        self.assertIsNotNone(content)

    @pytest.mark.asyncio
    async def test_function_call_with_clarify(self):
        functions = [
            {
                "name": "get_current_weather",
                "description": "获得指定地点的天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "省，市名，例如：河北省"},
                        "unit": {"type": "string", "enum": ["摄氏度", "华氏度"]},
                    },
                    "required": ["location"],
                },
                "responses": {
                    "type": "object",
                    "properties": {
                        "temperature": {"type": "number", "description": "当前温度"},
                        "weather_condition": {"type": "string", "description": "当前天气状况，例如：晴，多云，雨等"},
                        "humidity": {"type": "number", "description": "当前湿度百分比"},
                        "wind_speed": {"type": "number", "description": "风速，单位为公里每小时或英里每小时"},
                    },
                    "required": ["temperature", "weather_condition"],
                },
            }
        ]
        eb = ERNIEBot(
            model="ernie-3.5",
            api_type="aistudio",
            enable_human_clarify=True,
            enable_multi_step_tool_call=True,
        )
        messages = [
            HumanMessage(content="这个地方今天天气如何？"),
        ]
        res = await eb.chat(messages, functions=functions)

        self.assertTrue(isinstance(res, AIMessage))

        messages.append(res)
        messages.append(HumanMessage(content="深圳"))
        res_2 = await eb.chat(messages, functions=functions)
        self.assertTrue(hasattr(res_2, "function_call"))
        self.assertTrue(res_2.function_call["arguments"], '{"location":"深圳"}')
