from __future__ import annotations

import socket
import time
import unittest

import requests

from erniebot_agent.tools import RemoteToolkit
from erniebot_agent.tools.calculator_tool import CalculatorTool
from erniebot_agent.tools.current_time_tool import CurrentTimeTool
from erniebot_agent.tools.remote_tool import RemoteTool
from erniebot_agent.tools.tool_manager import ToolManager


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.01)
        try:
            s.bind(("localhost", port))
            return False
        except socket.error:
            return True


class TestToolManagerServe(unittest.IsolatedAsyncioTestCase):
    def avaliable_free_port(self, exclude=None):
        exclude = exclude or []
        for port in range(8000, 9000):
            if port in exclude:
                continue
            if is_port_in_use(port):
                continue
            return port

        raise ValueError("can not get valiable port in [8000, 8200]")

    def setUp(self) -> None:
        from threading import Thread

        self.port = self.avaliable_free_port()
        self.tool_manager = ToolManager([CurrentTimeTool(), CalculatorTool()])
        p = Thread(target=self.tool_manager.serve, args=(self.port,))
        p.daemon = True
        p.start()

    def wait_until_server_is_ready(self):
        while True:
            if is_port_in_use(self.port):
                break

            print("waiting for server ...")
            time.sleep(1)

    async def test_plugin_schema(self):
        self.wait_until_server_is_ready()

        # 1. get openapi yaml
        response = requests.get(f"http://127.0.0.1:{self.port}/.well-known/openapi.yaml")
        openapi = response.json()

        # 2. validate
        self.assertEqual(openapi["info"]["version"], "0.0")

        # 3. remote-toolkit
        toolkit = RemoteToolkit.from_url(f"http://127.0.0.1:{self.port}")
        self.assertEqual(len(toolkit.get_tools()), 2)

        # 4. current time description
        tool: RemoteTool = toolkit.get_tool("CurrentTimeTool")
        self.assertEqual(tool.tool_view.description, CurrentTimeTool.description)

        # 5. parameters
        self.assertIsNone(tool.tool_view.parameters)
        model_fields = tool.tool_view.returns.model_fields
        self.assertEqual(len(model_fields), 1)
        self.assertIn("current_time", model_fields)
        current_time = model_fields["current_time"]
        self.assertEqual(current_time.annotation, str)
        self.assertEqual(
            current_time.description, CurrentTimeTool.ouptut_type.model_fields["current_time"].description
        )


def test_tool_manager_crud():
    pass


def test_tool_manager_get_names():
    pass


def test_tool_manager_get_names_and_descriptions():
    pass


def test_tool_manager_get_schemas():
    pass
