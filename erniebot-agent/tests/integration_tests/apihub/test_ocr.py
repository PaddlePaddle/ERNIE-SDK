from __future__ import annotations

import asyncio

import pytest
from erniebot_agent.file_io.file_manager import FileManager
from erniebot_agent.tools.remote_toolkit import RemoteToolkit

from .base import RemoteToolTesting


class TestRemoteTool(RemoteToolTesting):
    def setUp(self) -> None:
        super().setUp()
        self.file_manager = FileManager()
        self.file = asyncio.run(
            self.file_manager.create_file_from_path(
                self.download_file(
                    "https://paddlenlp.bj.bcebos.com/ebagent/ci/fixtures/remote-tools/ocr_table.png"
                )
            )
        )

    @pytest.mark.asyncio
    async def test_ocr_general(self):
        toolkit = RemoteToolkit.from_aistudio("ocr-general")

        agent = self.get_agent(toolkit)

        result = await agent.async_run("请帮我提取一下这个图片中的内容", files=[self.file])
        self.assertEqual(len(result.files), 1)
        print(result)

    @pytest.mark.asyncio
    async def test_ocr_pp(self):
        toolkit = RemoteToolkit.from_aistudio("pp-structure-v2")

        agent = self.get_agent(toolkit)

        result = await agent.async_run("请帮我提取一下这个表格的内容", files=[self.file])
        self.assertEqual(len(result.files), 1)
        print(result)

    @pytest.mark.asyncio
    async def test_shopping_receipt(self):
        toolkit = RemoteToolkit.from_aistudio("shopping-receipt")

        agent = self.get_agent(toolkit)
        file = await self.file_manager.create_file_from_path(
            self.download_file(
                "https://paddlenlp.bj.bcebos.com/ebagent/ci/fixtures/remote-tools/xiaopiao.png"
            )
        )
        result = await agent.async_run("这张购物小票中有什么东西", files=[file])
        self.assertEqual(len(result.files), 1)
        print(result)

    @pytest.mark.asyncio
    async def test_formula(self):
        toolkit = RemoteToolkit.from_aistudio("formula")

        agent = self.get_agent(toolkit)
        file = await self.file_manager.create_file_from_path(
            self.download_file(
                "https://paddlenlp.bj.bcebos.com/ebagent/ci/fixtures/remote-tools/formula.png"
            )
        )
        result = await agent.async_run("请抽取一下这张图片里面的公式：", files=[file])
        self.assertEqual(len(result.files), 1)
        print(result)
