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
        files = self.get_files(result)
        action_steps = self.get_action_steps(result)
        self.assertGreater(len(action_steps), 0)
        self.assertEqual(len(files), 1)
        self.assertIn(".wav", files[0].filename)
