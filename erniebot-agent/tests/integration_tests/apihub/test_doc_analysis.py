from __future__ import annotations

import pytest

from erniebot_agent.tools.remote_toolkit import RemoteToolkit

from .base import RemoteToolTesting


class TestRemoteTool(RemoteToolTesting):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.file = await self.file_manager.create_file_from_path(self.download_fixture_file("城市管理执法办法.pdf"))

    @pytest.mark.asyncio
    async def test_doc_analysis(self):
        toolkit = RemoteToolkit.from_aistudio("doc-analysis", file_manager=self.file_manager)

        agent = self.get_agent(toolkit)

        result = await agent.run("请帮我分析一下这个文档里面的内容：", files=[self.file])
        self.assertEqual(len(result.files), 1)
        self.assertIn("城市管理执法办法", result.text)

    async def test_official_doc(self):
        toolkit = RemoteToolkit.from_aistudio("official-doc-rec", file_manager=self.file_manager)

        agent = self.get_agent(toolkit)

        result = await agent.run("帮我识别一下这个文件里面的内容：", files=[self.file])
        self.assertEqual(len(result.files), 1)
        self.assertIn("城市管理执法办法", result.text)
