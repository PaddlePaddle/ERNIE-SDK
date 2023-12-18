from __future__ import annotations

import pytest
from erniebot_agent.file_io import get_file_manager
from erniebot_agent.tools import RemoteToolkit

from .base import RemoteToolTesting


class TestRemoteTool(RemoteToolTesting):
    @pytest.mark.asyncio
    async def test_pp_tinypose(self):
        toolkit = RemoteToolkit.from_aistudio("pp-tinypose")
        tools = toolkit.get_tools()
        print(tools[0].function_call_schema())

        agent = self.get_agent(toolkit)

        file_manager = get_file_manager()
        file_path = self.download_file(
            "https://paddlenlp.bj.bcebos.com/ebagent/ci/fixtures/remote-tools/pp_tinypose_input_img.jpg"
        )
        file = await file_manager.create_file_from_path(file_path)

        result = await agent.async_run("检测这张图片中的人体关键点，包含的文件为：", files=[file])
        self.assertEqual(len(result.files), 2)
        self.assertEqual(len(result.actions), 1)
        self.assertIn("file-", result.text)
