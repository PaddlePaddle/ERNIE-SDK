import asyncio
import os

from erniebot_agent.chat_models.erniebot import ERNIEBot
from erniebot_agent.message import FunctionMessage, HumanMessage


async def test_function_call(model="ernie-bot"):
    api_type = "aistudio"
    access_token = os.getenv("ACCESS_TOKEN")  # set your access token as an environment variable
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

    eb = ERNIEBot(model=model, api_type=api_type, access_token=access_token)
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


if __name__ == "__main__":
    asyncio.run(test_function_call())
