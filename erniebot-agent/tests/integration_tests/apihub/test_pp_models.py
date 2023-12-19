from __future__ import annotations

import pytest
from erniebot_agent.file_io import get_file_manager
from erniebot_agent.tools.remote_toolkit import RemoteToolkit

from .base import RemoteToolTesting


class TestPPRemoteTool(RemoteToolTesting):
    @pytest.mark.asyncio
    async def test_pp_matting(self):
        toolkit = RemoteToolkit.from_aistudio("pp-matting")
        file_manager = get_file_manager()

        file = await file_manager.create_file_from_path(
            self.download_file("https://paddlenlp.bj.bcebos.com/ebagent/ci/fixtures/remote-tools/trans.png")
        )
        agent = self.get_agent(toolkit)

        result = await agent.async_run("请帮我对图片中的人像抠出来", files=[file])

        self.assertEqual(len(result.files), 2)

    @pytest.mark.asyncio
    async def test_pp_human_v2(self):
        toolkit = RemoteToolkit.from_aistudio("pp-human-v2")
        file_manager = get_file_manager()

        file = await file_manager.create_file_from_path(
            self.download_file(
                "https://paddlenlp.bj.bcebos.com/ebagent/ci/fixtures/remote-tools/human_attr.jpg"
            )
        )
        agent = self.get_agent(toolkit)

        result = await agent.async_run("请帮我分割图中的人", files=[file])
        self.assertEqual(len(result.files), 1)  # input的image不会出现
