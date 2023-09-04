import json

import erniebot
from erniebot.utils import logger


def test_function_calling(model="ernie-bot-3.5"):
    chat_completion = erniebot.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "百度公司当前在纳斯达克的股价是多少？",
            },
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": "get_current_price",
                    "arguments": json.dumps({
                        "company": "百度",
                        "exchange": "纳斯达克",
                    }),
                },
            },
            {
                "role": "function",
                "name": "get_current_price",
                "content": json.dumps({
                    "price": 146.47,
                    "unit": "美元",
                    "change": "上涨2.55%",
                }),
            },
        ],
        functions=[{
            "name": "get_current_price",
            "description": "获得指定公司的股价",
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {
                        "type": "string",
                        "description": "公司名，例如：腾讯，阿里巴巴",
                    },
                    "exchange": {
                        "type": "string",
                        "enum": ["纳斯达克", "上海证券交易所", "香港证券交易所"],
                    },
                },
                "required": ["company", "exchange"],
            },
            "responses": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "float",
                        "description": "当日股票价格",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["人民币", "美元", "港币"],
                        "description": "股票价格货币类型",
                    },
                    "change": {
                        "type": "string",
                        "description": "当日股票价格变化，如下跌3%，上涨0.5%",
                    },
                },
            },
        }, ],
        stream=False)
    print(chat_completion)


if __name__ == "__main__":
    logger.set_level("WARNING")
    erniebot.api_type = "qianfan"

    test_function_calling(model="ernie-bot-turbo")
