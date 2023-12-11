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


def create_chat_completion_with_plugins():
    response = erniebot.ChatCompletionWithPlugins.create(
        messages=[
            {
                "role": "user",
                "content": "帮我画一个饼状图：8月的用户反馈中，BUG有100条，需求有100条，使用咨询100条，总共300条反馈。",
            },
        ],
        plugins=["eChart"],
        stream=False,
    )
    print(response.get_result())


def create_chat_completion_with_plugins_stream():
    response = erniebot.ChatCompletionWithPlugins.create(
        messages=[
            {
                "role": "user",
                "content": "帮我画一个饼状图：8月的用户反馈中，BUG有100条，需求有100条，使用咨询100条，总共300条反馈。",
            },
        ],
        plugins=["eChart"],
        stream=True,
    )

    for item in response:
        sys.stdout.write(item.get_result())
        sys.stdout.flush()
    sys.stdout.write("\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    erniebot.api_type = "qianfan"

    create_chat_completion_with_plugins()

    create_chat_completion_with_plugins_stream()
