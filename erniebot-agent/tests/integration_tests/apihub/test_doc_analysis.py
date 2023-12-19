from __future__ import annotations

import asyncio

import pytest
from erniebot_agent.file_io import get_file_manager
from erniebot_agent.tools.remote_toolkit import RemoteToolkit

from .base import RemoteToolTesting


class TestRemoteTool(RemoteToolTesting):
    def setUp(self) -> None:
        super().setUp()
        self.file_manager = get_file_manager()
        self.file = asyncio.run(
            self.file_manager.create_file_from_path(
                self.download_file(
                    "https://paddlenlp.bj.bcebos.com/ebagent/ci/fixtures/remote-tools/城市管理执法办法.pdf"
                )
            )
        )

    @pytest.mark.asyncio
    async def test_doc_analysis(self):
        toolkit = RemoteToolkit.from_aistudio("doc-analysis")

        agent = self.get_agent(toolkit)

        result = await agent.async_run("请帮我分析一下这个文档里面的内容：", files=[self.file])
        self.assertEqual(len(result.files), 1)
        self.assertGreater(len(result.text), 5)

    async def test_official_doc(self):
        toolkit = RemoteToolkit.from_aistudio("official-doc-rec")

        agent = self.get_agent(toolkit)

        result = await agent.async_run("帮我识别一下这个文件里面的内容：", files=[self.file])
        self.assertEqual(len(result.files), 1)
        self.assertGreater(len(result.text), 5)
