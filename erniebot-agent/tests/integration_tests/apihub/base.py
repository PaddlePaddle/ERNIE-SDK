from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from typing import Optional

import requests
from erniebot_agent.agents.functional_agent import FunctionalAgent
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.memory import WholeMemory
from erniebot_agent.tools.remote_toolkit import RemoteToolkit
from erniebot_agent.tools.tool_manager import ToolManager


class RemoteToolTesting(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def download_file(self, url, file_name: Optional[str] = None):
        image_response = requests.get(url)
        if file_name is None:
            file_name = os.path.basename(url)

        path = os.path.join(self.temp_dir, file_name)
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
