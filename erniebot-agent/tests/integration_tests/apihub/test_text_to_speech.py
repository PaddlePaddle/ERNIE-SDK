from __future__ import annotations

import pytest

from erniebot_agent.tools.remote_toolkit import RemoteToolkit

from .base import RemoteToolTesting


class TestRemoteTool(RemoteToolTesting):
    @pytest.mark.asyncio
    async def test_text_to_speech(self):
        toolkit = RemoteToolkit.from_aistudio("texttospeech", file_manager=self.file_manager)

        agent = self.get_agent(toolkit)

        result = await agent.run("请把：“我爱中国”转化为语音")
        self.assertGreater(len(result.actions), 0)
        self.assertEqual(len(result.files), 1)
        self.assertIn(".wav", result.files[0].file.filename)
