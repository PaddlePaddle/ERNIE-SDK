from __future__ import annotations

import pytest

from erniebot_agent.tools.remote_toolkit import RemoteToolkit

from .base import RemoteToolTesting


# 未通过
class TestRemoteTool(RemoteToolTesting):
    @pytest.mark.asyncio
    async def test_paddle_tts(self):
        toolkit = RemoteToolkit.from_aistudio("12140")
        agent = self.get_agent(toolkit)

        result = await agent.async_run("帮我把：“我爱中国”转化成语音")
        self.assertEqual(len(result.files), 1)
        assert ".wav" in result.files[0].file.filename

    # @pytest.mark.asyncio
    # async def test_paddle_tts(self):
    #     url = "http://tool-12140.sandbox-aistudio-hub.baidu.com"
    #     toolkit = RemoteToolkit.from_url(url, access_token="1dc43e5843cfb51b7b41ba766aff2372cf2f3ccb")

    #     agent = self.get_agent(toolkit)
    #     result = await agent.async_run(f"帮我把：“我爱中国”转化成语音")

    #     print(result)

    #     assert len(result.files) == 1
    #     assert ".wav" in result.files[0].file.filename
    #     file = result.files[0].file

    #     url = "http://tool-asr.sandbox-aistudio-hub.baidu.com"

    #     content = await file.read_contents()
