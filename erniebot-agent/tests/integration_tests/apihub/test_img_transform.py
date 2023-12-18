from __future__ import annotations

import asyncio

import pytest
from erniebot_agent.file_io.file_manager import FileManager
from erniebot_agent.tools.remote_toolkit import RemoteToolkit
from PIL import Image

from .base import RemoteToolTesting


class TestRemoteTool(RemoteToolTesting):
    def setUp(self) -> None:
        super().setUp()
        self.file_manager = FileManager()
        self.file = asyncio.run(
            self.file_manager.create_file_from_path(
                self.download_file(
                    "https://paddlenlp.bj.bcebos.com/ebagent/ci/fixtures/remote-tools/trans.png"
                )
            )
        )

    @pytest.mark.asyncio
    async def test_img_style_trans(self):
        toolkit = RemoteToolkit.from_aistudio("img-style-trans")

        agent = self.get_agent(toolkit)

        result = await agent.async_run("帮我把这个图片转换为铅笔风格", files=[self.file])
        self.assertEqual(len(result.files), 2)
        Image.open(result.files[-1].file.path).show()

    @pytest.mark.asyncio
    async def test_person_animation(self):
        toolkit = RemoteToolkit.from_aistudio("person-animation")

        agent = self.get_agent(toolkit)

        result = await agent.async_run("帮我把这张人像图片转化为动漫的图片", files=[self.file])
        self.assertEqual(len(result.files), 2)
        Image.open(result.files[-1].file.path).show()
