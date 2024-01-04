from __future__ import annotations

import pytest

from erniebot_agent.tools.remote_toolkit import RemoteToolkit

from .base import RemoteToolTesting


class TestRemoteTool(RemoteToolTesting):
    @pytest.mark.asyncio
    async def test_tool(self):
        toolkit = RemoteToolkit.from_aistudio("text-moderation", file_manager=self.file_manager)
        agent = self.get_agent(toolkit)

        result = await agent.run("请判断：“我爱我的家乡” 这句话是否合规")
        action_steps = self.get_action_steps(result)
        self.assertGreater(len(action_steps), 0)
        self.assertIn("合规", result.text)
