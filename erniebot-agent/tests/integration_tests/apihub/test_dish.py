from __future__ import annotations

import pytest
from erniebot_agent.file_io import get_file_manager
from erniebot_agent.tools.remote_toolkit import RemoteToolkit

from .base import RemoteToolTesting


class TestRemoteTool(RemoteToolTesting):
    @pytest.mark.asyncio
    async def test_dish_classify(self):
        toolkit = RemoteToolkit.from_aistudio("dish-classify")

        file_manager = get_file_manager()

        file = await file_manager.create_file_from_path(
            self.download_file("https://paddlenlp.bj.bcebos.com/ebagent/ci/fixtures/remote-tools/dish.png")
        )
        agent = self.get_agent(toolkit)

        result = await agent.async_run("这张照片中的菜品是什么", files=[file])
        self.assertEqual(len(result.files), 1)
        self.assertIn("蛋", result.text)
