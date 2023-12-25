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

import erniebot


def test_function_calling():
    response = erniebot.ChatCompletion.create(
        model="ernie-3.5",
        messages=[
            {
                "role": "user",
                "content": "深圳市今天气温多少摄氏度？",
            },
        ],
        functions=[
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
            },
        ],
        stream=False,
    )
    print(response.get_result())


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    erniebot.api_type = "qianfan"

    test_function_calling()
