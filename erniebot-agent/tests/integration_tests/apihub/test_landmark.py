from __future__ import annotations

import pytest
from erniebot_agent.file_io.file_manager import FileManager
from erniebot_agent.tools.remote_toolkit import RemoteToolkit

from .base import RemoteToolTesting


class TestRemoteTool(RemoteToolTesting):
    @pytest.mark.asyncio
    async def test_dish_classify(self):
        toolkit = RemoteToolkit.from_aistudio("landmark-rec")

        file_manager = FileManager()

        file = await file_manager.create_file_from_path(
            self.download_file(
                "https://paddlenlp.bj.bcebos.com/ebagent/ci/fixtures/remote-tools"
                "/shanghai-dongfangmingzhu.png"
            )
        )
        agent = self.get_agent(toolkit)

        result = await agent.async_run("这张照片中的地标是什么", files=[file])
        self.assertEqual(len(result.files), 1)
        print(result)
