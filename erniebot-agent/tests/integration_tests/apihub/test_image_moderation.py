from __future__ import annotations

import pytest

from erniebot_agent.tools.remote_toolkit import RemoteToolkit

from .base import RemoteToolTesting


class TestRemoteTool(RemoteToolTesting):
    @pytest.mark.asyncio
    async def test_tool(self):
        toolkit = RemoteToolkit.from_aistudio("image-moderation", file_manager=self.file_manager)

        file = await self.file_manager.create_file_from_path(self.download_fixture_file("shouxiezi.png"))
        agent = self.get_agent(toolkit)

        result = await agent.run("请判断这张图片是否合规", files=[file])
        action_steps = self.get_action_steps(result)
        self.assertGreater(len(action_steps), 0)
        self.assertIn("合规", result.text)
