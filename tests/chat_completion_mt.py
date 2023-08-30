#!/usr/bin/env python

import threading

import erniebot
from erniebot.utils import logger

NUM_TASKS = 4


def create_chat_completion(model):
    resp = erniebot.ChatCompletion.create(
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
    print(resp)


def create_chat_completion_stream(model):
    resp = erniebot.ChatCompletion.create(
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

    for item in resp:
        print(item)


def test_mt(target, args):
    threads = []
    for _ in range(NUM_TASKS):
        thread = threading.Thread(target=target, args=args)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == '__main__':
    logger.set_level("WARNING")
    erniebot.api_type = 'qianfan'

    # 批量返回
    test_mt(create_chat_completion, args=('ernie-bot-turbo', ))

    # 流式逐句返回
    test_mt(create_chat_completion_stream, args=('ernie-bot-turbo', ))
