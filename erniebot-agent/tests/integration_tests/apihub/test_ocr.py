from __future__ import annotations

import json

import pytest

from erniebot_agent.tools.remote_toolkit import RemoteToolkit

from .base import RemoteToolTesting


class TestRemoteTool(RemoteToolTesting):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.file = await self.file_manager.create_file_from_path(
            self.download_fixture_file("ocr_table.png")
        )

    @pytest.mark.asyncio
    async def test_ocr_general(self):
        toolkit = RemoteToolkit.from_aistudio("ocr-general", file_manager=self.file_manager)

        agent = self.get_agent(toolkit)

        result = await agent.run("请帮我提取一下这个图片中的内容", files=[self.file])
        self.assertEqual(len(result.files), 1)
        self.assertIn("表格", result.text)

    @pytest.mark.asyncio
    async def test_ocr_pp(self):
        toolkit = RemoteToolkit.from_aistudio("pp-structure-v2", file_manager=self.file_manager)

        agent = self.get_agent(toolkit)

        result = await agent.run("请帮我提取一下这个表格的内容", files=[self.file])
        self.assertEqual(len(result.files), 1)
        self.assertIn("设备", result.text)

    @pytest.mark.asyncio
    async def test_pp_ocr_v4(self):
        toolkit = RemoteToolkit.from_aistudio("pp-ocrv4", file_manager=self.file_manager)

        file = await self.file_manager.create_file_from_path(
            self.download_fixture_file("ocr_example_input.png")
        )
        agent = self.get_agent(toolkit)

        response = await agent.run("请帮我识别出这幅图片中的文字", files=[file])

        self.assertEqual(len(response.steps), 1)
        decoded_tool_ret = json.loads(response.chat_history[2].content)
        self.assertEqual(decoded_tool_ret, {"result": "中国\n汉字"})

    @pytest.mark.asyncio
    async def test_shopping_receipt(self):
        toolkit = RemoteToolkit.from_aistudio("shopping-receipt", file_manager=self.file_manager)

        agent = self.get_agent(toolkit)
        file = await self.file_manager.create_file_from_path(self.download_fixture_file("xiaopiao.png"))
        result = await agent.run("这张购物小票中有什么东西", files=[file])
        self.assertEqual(len(result.files), 1)
        self.assertGreater(len(result.text), 5)

    @pytest.mark.asyncio
    async def test_formula(self):
        toolkit = RemoteToolkit.from_aistudio("formula", file_manager=self.file_manager)

        agent = self.get_agent(toolkit)
        file = await self.file_manager.create_file_from_path(self.download_fixture_file("fomula.png"))
        result = await agent.run("请抽取一下这张图片里面的公式：", files=[file])
        self.assertEqual(len(result.files), 1)
        self.assertGreater(len(result.text), 5)
