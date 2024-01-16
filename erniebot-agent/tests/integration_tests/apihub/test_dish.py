from __future__ import annotations

import pytest

from erniebot_agent.tools.remote_toolkit import RemoteToolkit

from .base import RemoteToolTesting


class TestRemoteTool(RemoteToolTesting):
    @pytest.mark.asyncio
    async def test_dish_classify(self):
        toolkit = RemoteToolkit.from_aistudio("dish-classify", file_manager=self.file_manager)

        file = await self.file_manager.create_file_from_path(self.download_fixture_file("dish.png"))
        agent = self.get_agent(toolkit)

        result = await agent.run("这张照片中的菜品是什么", files=[file])
        files = self.get_files(result)
        self.assertEqual(len(files), 1)
        self.assertIn("蛋", result.text)
