from __future__ import annotations

import pytest

from erniebot_agent.tools.remote_toolkit import RemoteToolkit

from .base import RemoteToolTesting


class TestRemoteTool(RemoteToolTesting):
    @pytest.mark.asyncio
    async def test_car_classify(self):
        toolkit = RemoteToolkit.from_aistudio("car-classify", file_manager=self.file_manager)

        file = await self.file_manager.create_file_from_path(self.download_fixture_file("biyadi.png"))
        agent = self.get_agent(toolkit)

        result = await agent.run("这张照片中 车是什么牌子的车", files=[file])
        self.assertEqual(len(result.files), 1)
        self.assertIn("比亚迪", result.text)
