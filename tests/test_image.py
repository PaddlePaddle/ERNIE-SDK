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

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    erniebot.api_type = "yinian"

    response = erniebot.Image.create(
        model="ernie-vilg-v2", prompt="请帮我画一只开心的袋熊", width=512, height=512, version="v2", image_num=1
    )
    print(response.get_result())
