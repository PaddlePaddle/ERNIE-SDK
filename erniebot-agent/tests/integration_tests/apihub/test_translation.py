from __future__ import annotations

import pytest

from erniebot_agent.tools.remote_toolkit import RemoteToolkit

from .base import RemoteToolTesting


class TestRemoteTool(RemoteToolTesting):
    @pytest.mark.asyncio
    async def test_tool(self):
        toolkit = RemoteToolkit.from_aistudio("translation", file_manager=self.file_manager)

        agent = self.get_agent(toolkit)

        result = await agent.run("请将：”我爱中国“ 翻译成英语")
        action_steps = self.get_action_steps(result)
        self.assertGreater(len(action_steps), 0)
        text = result.text.lower()
        self.assertIn("i love china", text)
