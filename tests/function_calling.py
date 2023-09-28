import json

import erniebot
from erniebot.utils import logger


def test_function_calling(model="ernie-bot"):
    response = erniebot.ChatCompletion.create(
        model=model,
        messages=[{
            "role": "user",
            "content": "深圳市今天气温多少摄氏度？",
        }, ],
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
    print(response.get_result())


if __name__ == "__main__":
    logger.set_level("WARNING")
    erniebot.api_type = "qianfan"

    test_function_calling(model="ernie-bot")
