from __future__ import annotations

import pytest

from erniebot_agent.tools.remote_toolkit import RemoteToolkit

from .base import RemoteToolTesting


class TestRemoteTool(RemoteToolTesting):
    @pytest.mark.asyncio
    async def test_doc_rm_bnd(self):
        toolkit = RemoteToolkit.from_aistudio("rm-doc-img-bnd", file_manager=self.file_manager)

        file = await self.file_manager.create_file_from_path(self.download_fixture_file("biyadi.png"))
        agent = self.get_agent(toolkit)

        result = await agent.run("请帮我把这张图片的底纹给删掉", files=[file])
        self.assertEqual(len(result.files), 2)
