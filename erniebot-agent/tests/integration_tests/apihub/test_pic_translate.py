from __future__ import annotations

import pytest

from erniebot_agent.tools.remote_toolkit import RemoteToolkit

from .base import RemoteToolTesting


class TestRemoteTool(RemoteToolTesting):

    @pytest.mark.asyncio
    async def test_tool(self):
        toolkit = RemoteToolkit.from_aistudio("pic-translate")

        file = await self.file_manager.create_file_from_path(self.download_fixture_file("shouxiezi.png"))
        agent = self.get_agent(toolkit)

        result = await agent.async_run("这张照片里面讲了啥？", files=[file])
        self.assertGreater(len(result.actions), 0)
        # self.assertIn("合规", result.text)
