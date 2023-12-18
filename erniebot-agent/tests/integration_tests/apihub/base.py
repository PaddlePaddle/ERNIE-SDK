from __future__ import annotations

import os
import tempfile
import unittest

import requests
from erniebot_agent.agents.functional_agent import FunctionalAgent
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.memory import WholeMemory
from erniebot_agent.tools.base import RemoteToolkit
from erniebot_agent.tools.tool_manager import ToolManager


class RemoteToolTesting(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def get_image(self):
        image_response = requests.get(
            "https://ai-studio-static-online.cdn.bcebos.com/"
            "dcdfa7f8c35f4d5f9e0eeab7e590f5f4b576bb1728e94bb4a889b34d833397d2"
        )
        path = os.path.join(self.temp_dir, "test.png")
        with open(path, "wb") as f:
            f.write(image_response.content)

        return path

    def get_agent(self, toolkit: RemoteToolkit):
        if "EB_BASE_URL" in os.environ:
            llm = ERNIEBot(model="ernie-bot", api_type="custom")
        else:
            llm = ERNIEBot(model="ernie-bot")

        return FunctionalAgent(
            llm=llm,
            tools=ToolManager(tools=toolkit.get_tools()),
            memory=WholeMemory(),
        )
