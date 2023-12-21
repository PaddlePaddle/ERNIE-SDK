from __future__ import annotations

import unittest

from erniebot_agent.tools.current_time_tool import CurrentTimeTool


class TestCalculatorTool(unittest.TestCase):
    def test_schema(self):
        calculator = CurrentTimeTool()
        function_call_schema = calculator.function_call_schema()

        self.assertEqual(function_call_schema["description"], CurrentTimeTool.description)
        self.assertEqual(
            function_call_schema["responses"]["properties"]["current_time"]["type"],
            "string",
        )