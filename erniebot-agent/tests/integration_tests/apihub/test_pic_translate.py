from __future__ import annotations

import pytest

from erniebot_agent.tools.remote_toolkit import RemoteToolkit

from .base import RemoteToolTesting


class TestRemoteTool(RemoteToolTesting):
    @pytest.mark.asyncio
    async def test_tool(self):
        toolkit = RemoteToolkit.from_aistudio("pic-translate", file_manager=self.file_manager)

        file = await self.file_manager.create_file_from_path(self.download_fixture_file("shouxiezi.png"))
        agent = self.get_agent(toolkit)

        result = await agent.run("这张照片里面讲了啥？", files=[file])
        action_steps = self.get_action_steps(result)
        self.assertEqual(len(action_steps), 1)
        # 润色不太稳定
        self.assertIn("春天", result.text)
        self.assertIn("梦", result.text)
