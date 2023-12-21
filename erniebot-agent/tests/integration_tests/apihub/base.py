from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from typing import Optional

import requests

from erniebot_agent.agents.functional_agent import FunctionalAgent
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.file_io.file_manager import FileManager
from erniebot_agent.memory import WholeMemory
from erniebot_agent.tools import RemoteToolkit
from erniebot_agent.tools.tool_manager import ToolManager


class RemoteToolTesting(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.file_manager = FileManager()

    async def asyncTearDown(self) -> None:
        shutil.rmtree(self.temp_dir)
        await self.file_manager.close()

    def download_file(self, url, file_name: Optional[str] = None):
        image_response = requests.get(url)
        if file_name is None:
            file_name = os.path.basename(url)

        path = os.path.join(self.temp_dir, file_name)
        with open(path, "wb") as f:
            f.write(image_response.content)

        return path

    def download_fixture_file(self, file_name: str):
        url = f"https://paddlenlp.bj.bcebos.com/ebagent/ci/fixtures/remote-tools/{file_name}"
        return self.download_file(url)

    def get_agent(self, toolkit: RemoteToolkit):
        if "EB_BASE_URL" in os.environ:
            llm = ERNIEBot(model="ernie-3.5", api_type="custom")
        else:
            llm = ERNIEBot(model="ernie-3.5")

        return FunctionalAgent(
            llm=llm,
            tools=ToolManager(tools=toolkit.get_tools()),
            memory=WholeMemory(),
        )
