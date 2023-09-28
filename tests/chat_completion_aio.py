#!/usr/bin/env python

import asyncio
import sys

import erniebot
from erniebot.utils import logger

NUM_TASKS = 4


async def acreate_chat_completion(model):
    resp = await erniebot.ChatCompletion.acreate(
        model=model,
        messages=[{
            'role': 'user',
            'content': "请问你是谁？"
        }, {
            'role': 'assistant',
            'content':
            "我是百度公司开发的人工智能语言模型，我的中文名是文心一言，英文名是ERNIE-Bot，可以协助您完成范围广泛的任务并提供有关各种主题的信息，比如回答问题，提供定义和解释及建议。如果您有任何问题，请随时向我提问。"
        }, {
            'role': 'user',
            'content': "我在深圳，周末可以去哪里玩？"
        }],
        stream=False)
    print(resp.get_result())


async def acreate_chat_completion_stream(model):
    resp = await erniebot.ChatCompletion.acreate(
        model=model,
        messages=[{
            'role': 'user',
            'content': "请问你是谁？"
        }, {
            'role': 'assistant',
            'content':
            "我是百度公司开发的人工智能语言模型，我的中文名是文心一言，英文名是ERNIE-Bot，可以协助您完成范围广泛的任务并提供有关各种主题的信息，比如回答问题，提供定义和解释及建议。如果您有任何问题，请随时向我提问。"
        }, {
            'role': 'user',
            'content': "我在深圳，周末可以去哪里玩？"
        }],
        stream=True)

    async for item in resp:
        sys.stdout.write(item.get_result())
        sys.stdout.flush()
    sys.stdout.write("\n")


async def test_aio(target, args):
    coroutines = []
    for _ in range(NUM_TASKS):
        coroutine = target(*args)
        coroutines.append(coroutine)
    await asyncio.gather(*coroutines)


if __name__ == '__main__':
    logger.set_level("WARNING")
    erniebot.api_type = 'qianfan'

    # 批量返回
    asyncio.run(test_aio(acreate_chat_completion, args=('ernie-bot-turbo', )))

    # 流式逐句返回
    asyncio.run(
        test_aio(
            acreate_chat_completion_stream, args=('ernie-bot-turbo', )))
