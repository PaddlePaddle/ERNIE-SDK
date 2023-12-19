from __future__ import annotations

import json

import pytest
from erniebot_agent.file_io.file_manager import FileManager
from erniebot_agent.messages import FunctionMessage
from erniebot_agent.tools.remote_toolkit import RemoteToolkit

from .base import RemoteToolTesting


class TestVehicleAnalysis(RemoteToolTesting):
    @pytest.mark.asyncio
    async def test_analyze_vehicle(self):
        toolkit = RemoteToolkit.from_aistudio("vehicle-analysis")

        file_manager = FileManager()

        file = await file_manager.create_file_from_path(
            self.download_file(
                "https://paddlenlp.bj.bcebos.com/ebagent/ci/fixtures/remote-tools/vehicle.jpg"
            )
        )
        agent = self.get_agent(toolkit)

        response = await agent.async_run("请帮我对这幅图进行交通场景分析", files=[file])

        expected_tool_name = toolkit.get_tool("analyzeVehicles").tool_name
        self.assertEqual(len(response.actions), 1)
        self.assertEqual(response.actions[0].tool_name, expected_tool_name)
        self.assertEqual(len(response.chat_history), 4)
        self.assertIsInstance(response.chat_history[2], FunctionMessage)
        self.assertEqual(response.chat_history[2].name, expected_tool_name)
        decoded_tool_ret = json.loads(response.chat_history[2].content)
        self.assertIn("vehicle_plates", decoded_tool_ret)
        self.assertEqual(decoded_tool_ret["vehicle_plates"], ["CCL9542"])
        self.assertIn("vehicle_attrs", decoded_tool_ret)
        self.assertEqual(decoded_tool_ret["vehicle_attrs"], [{"color": "blue", "kind": "Unknown"}])
        self.assertEqual(len(response.files), 2)
