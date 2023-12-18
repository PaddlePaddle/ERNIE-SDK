from __future__ import annotations

import pytest
from erniebot_agent.file_io.file_manager import FileManager
from erniebot_agent.tools.remote_toolkit import RemoteToolkit
from PIL import Image

from .base import RemoteToolTesting


class TestPPRemoteTool(RemoteToolTesting):
    @pytest.mark.asyncio
    async def test_pp_matting(self):
        toolkit = RemoteToolkit.from_aistudio("pp-matting")
        file_manager = FileManager()

        file = await file_manager.create_file_from_path(
            self.download_file("https://paddlenlp.bj.bcebos.com/ebagent/ci/fixtures/remote-tools/trans.png")
        )
        agent = self.get_agent(toolkit)

        result = await agent.async_run("请帮我对图片中的人像抠出来", files=[file])
        self.assertEqual(len(result.files), 2)
        Image.open(result.files[-1].file.path).show()

    @pytest.mark.asyncio
    async def test_pp_human_v2(self):
        toolkit = RemoteToolkit.from_aistudio("pp-human-v2")
        file_manager = FileManager()

        file = await file_manager.create_file_from_path(
            self.download_file(
                "https://paddlenlp.bj.bcebos.com/ebagent/ci/fixtures/remote-tools/human_attr.jpg"
            )
        )
        agent = self.get_agent(toolkit)

        result = await agent.async_run("请识别图中的几个行人", files=[file])
        self.assertEqual(len(result.files), 2)
        Image.open(result.files[-1].file.path).show()
