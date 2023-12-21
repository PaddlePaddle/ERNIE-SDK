from __future__ import annotations

import pytest

from erniebot_agent.tools.remote_toolkit import RemoteToolkit

from .base import RemoteToolTesting


class TestRemoteTool(RemoteToolTesting):

    @pytest.mark.asyncio
    async def test_tool(self):
        toolkit = RemoteToolkit.from_aistudio("highacc-ocr")

        file = await self.file_manager.create_file_from_path(self.download_fixture_file("shouxiezi.png"))
        agent = self.get_agent(toolkit)

        result = await agent.async_run("帮我看看这张照片中的内容有什么", files=[file])
        self.assertGreater(len(result.actions), 0)
        self.assertIn("春天的梦", result.text)
