import json

import erniebot
from erniebot.utils import logger


def test_function_calling(model="ernie-bot"):
    response = erniebot.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "深圳市今天气温如何？",
            },
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": "get_current_temperature",
                    "arguments": json.dumps(
                        {
                            "location": "深圳市",
                            "unit": "摄氏度",
                        },
                        ensure_ascii=False),
                },
            },
            {
                "role": "function",
                "name": "get_current_temperature",
                "content": json.dumps(
                    {
                        "temperature": 25,
                        "type": "摄氏度",
                    }, ensure_ascii=False),
            },
        ],
        functions=[{
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
                        "enum": ["摄氏度", "华氏度"],
                    },
                },
                "required": ["location", "unit"],
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
                        "enum": ["摄氏度", "华氏度"],
                    },
                },
            },
        }, ],
        stream=False)
    print(response)


if __name__ == "__main__":
    logger.set_level("WARNING")
    erniebot.api_type = "qianfan"

    test_function_calling(model="ernie-bot-turbo")
