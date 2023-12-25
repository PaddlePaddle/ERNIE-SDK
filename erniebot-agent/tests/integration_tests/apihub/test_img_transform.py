from __future__ import annotations

import pytest

from erniebot_agent.tools.remote_toolkit import RemoteToolkit

from .base import RemoteToolTesting


class TestRemoteTool(RemoteToolTesting):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.file = await self.file_manager.create_file_from_path(self.download_fixture_file("trans.png"))

    @pytest.mark.asyncio
    async def test_img_style_trans(self):
        toolkit = RemoteToolkit.from_aistudio("img-style-trans", file_manager=self.file_manager)

        agent = self.get_agent(toolkit)

        result = await agent.async_run("帮我把这个图片转换为卡通风格", files=[self.file])
        self.assertEqual(len(result.files), 2)

    @pytest.mark.asyncio
    async def test_person_animation(self):
        toolkit = RemoteToolkit.from_aistudio("person-animation", file_manager=self.file_manager)

        agent = self.get_agent(toolkit)

        result = await agent.async_run("帮我把这张人像图片转化为动漫的图片", files=[self.file])
        self.assertEqual(len(result.files), 2)
