#!/usr/bin/env python

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import sys

import erniebot


def create_chat_completion(model):
    response = erniebot.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "user", "content": "请问你是谁？"},
            {
                "role": "assistant",
                "content": (
                    "我是百度公司开发的人工智能语言模型，我的中文名是文心一言，英文名是ERNIE-Bot，可以协助您完成范围广泛的任务并提供有关各种主题的信息，"
                    "比如回答问题，提供定义和解释及建议。如果您有任何问题，请随时向我提问。"
                ),
            },
            {"role": "user", "content": "我在深圳，周末可以去哪里玩？"},
        ],
        stream=False,
    )
    print(response.get_result())


def create_chat_completion_json_mode(model):
    response = erniebot.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "user", "content": "文心一言是哪个公司开发的？"},
        ],
        stream=False,
        response_format="json_object",
    )
    print(response.get_result())


def create_chat_completion_stream(model):
    response = erniebot.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "user", "content": "请问你是谁？"},
            {
                "role": "assistant",
                "content": (
                    "我是百度公司开发的人工智能语言模型，我的中文名是文心一言，英文名是ERNIE-Bot，可以协助您完成范围广泛的任务并提供有关各种主题的信息，"
                    "比如回答问题，提供定义和解释及建议。如果您有任何问题，请随时向我提问。"
                ),
            },
            {"role": "user", "content": "我在深圳，周末可以去哪里玩？"},
        ],
        stream=True,
    )

    for item in response:
        sys.stdout.write(item.get_result())
        sys.stdout.flush()
    sys.stdout.write("\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    erniebot.api_type = "qianfan"

    create_chat_completion(model="ernie-turbo")
    create_chat_completion_stream(model="ernie-turbo")
    create_chat_completion_json_mode(model="ernie-lite")
