from __future__ import annotations

import pytest

from erniebot_agent.tools.remote_toolkit import RemoteToolkit

from .base import RemoteToolTesting


class TestRemoteTool(RemoteToolTesting):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.file = await self.file_manager.create_file_from_path(
            self.download_fixture_file("shouxiezi.png")
        )

    @pytest.mark.asyncio
    async def test_hand_text_rec(self):
        toolkit = RemoteToolkit.from_aistudio("hand-text-rec", file_manager=self.file_manager)

        agent = self.get_agent(toolkit)

        result = await agent.async_run("这张照片中的手写字是什么？", files=[self.file])
        self.assertEqual(len(result.files), 1)
        self.assertIn("春天的梦", result.text)
