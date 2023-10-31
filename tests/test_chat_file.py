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

import erniebot
from erniebot.utils.logging import logger

if __name__ == "__main__":
    logger.set_level("WARNING")

    erniebot.api_type = "qianfan"

    response = erniebot.ChatFile.create(
        messages=[
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，我是文心一言"},
            {
                "role": "user",
                "content": "<file>AI_Frameworks.pdf</file><url>https://ernie-bot-sdk.bj.bcebos.com/AI_Frameworks.pdf</url>",
            },
            {
                "role": "assistant",
                "content": "以下是该文档的简要内容：AI框架技术不断发展，已形成较为完整的体系，包括基础层、组件层和生态层。全球AI框架繁荣发展，多元化竞合态势逐渐显现，赋能学术科研和产业应用。为了应对未来多样化挑战，AI框架有六大技术趋势。虽然AI框架生态远未成熟，但未来发展空间广阔。",
            },
            {"role": "user", "content": "应对未来的技术挑战，AI框架有哪些独特的技术特色？"},
        ],
    )
    print(response.get_result())
