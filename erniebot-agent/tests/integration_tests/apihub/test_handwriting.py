from __future__ import annotations

import asyncio

import pytest

from erniebot_agent.tools.remote_toolkit import RemoteToolkit

from .base import RemoteToolTesting


class TestRemoteTool(RemoteToolTesting):
    def setUp(self) -> None:
        super().setUp()
        self.file = asyncio.run(
            self.file_manager.create_file_from_path(self.download_fixture_file("shouxiezi.png"))
        )

    @pytest.mark.asyncio
    async def test_hand_text_rec(self):
        toolkit = RemoteToolkit.from_aistudio("hand-text-rec")

        agent = self.get_agent(toolkit)

        result = await agent.async_run("这张照片中的手写字是什么？", files=[self.file])
        self.assertEqual(len(result.files), 1)
        self.assertIn("春天的梦", result.text)
