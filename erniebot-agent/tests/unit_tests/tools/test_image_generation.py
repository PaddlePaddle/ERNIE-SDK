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

import os
import unittest

import pytest
from erniebot_agent.agents.functional_agent import FunctionalAgent
from erniebot_agent.chat_models.erniebot import ERNIEBot
from erniebot_agent.memory.whole_memory import WholeMemory
from erniebot_agent.tools.image_generation_tool import ImageGenerationTool


class TestImageGenerationTool(unittest.TestCase):
    def test_schema(self):
        img_gen_tool = ImageGenerationTool(yinian_ak="xxx", yinian_sk="xxx")
        function_call_schema = img_gen_tool.function_call_schema()

        self.assertEqual(function_call_schema["description"], ImageGenerationTool.description)
        self.assertIn("parameters", function_call_schema)
        self.assertIn("responses", function_call_schema)

    @pytest.mark.asyncio
    async def test_tool(self):
        yinian_ak = os.environ.get("YINIAN_AK")
        yinian_sk = os.environ.get("YINIAN_SK")
        if yinian_ak is None or yinian_sk is None:
            return

        img_gen_tool = ImageGenerationTool(yinian_ak=yinian_ak, yinian_sk=yinian_sk)
        result = await img_gen_tool(prompt="画一只老虎", width=512, height=512)
        self.assertIn("image_path", result)

        for path in result["image_path"]:
            self.assertTrue(os.path.exists(path))

    @pytest.mark.asyncio
    async def test_agent(self):
        aistudio_access_token = os.environ["AISTUDIO_ACCESS_TOKEN"]
        if aistudio_access_token is None:
            return

        yinian_ak = os.environ.get("YINIAN_AK")
        yinian_sk = os.environ.get("YINIAN_SK")
        if yinian_ak is None or yinian_sk is None:
            return

        eb = ERNIEBot(model="ernie-bot", api_type="aistudio", access_token=aistudio_access_token)
        memory = WholeMemory()
        img_gen_tool = ImageGenerationTool(yinian_ak=yinian_ak, yinian_sk=yinian_sk)
        agent = FunctionalAgent(llm=eb, tools=[img_gen_tool], memory=memory)
        result = await agent.async_run("画1张小狗的图片")

        self.assertIn("image_path", result)
        for path in result["image_path"]:
            self.assertTrue(os.path.exists(path))
