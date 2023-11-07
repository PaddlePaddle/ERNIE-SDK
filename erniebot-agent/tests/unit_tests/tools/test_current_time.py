from __future__ import annotations

import unittest
from datetime import datetime

import pytest
from erniebot_agent.tools.current_time_tool import CurrentTimeTool


class TestCalculatorTool(unittest.TestCase):
    @pytest.mark.asyncio
    async def test_call(self):
        calculator = CurrentTimeTool()
        now = datetime.now()
        tool_result = await calculator()
        self.assertIn(f"{now.year}年", tool_result)
        self.assertIn(f"{now.month}月", tool_result)

    def test_schema(self):
        calculator = CurrentTimeTool()
        function_call_schema = calculator.function_call_schema()

        self.assertEqual(function_call_schema["description"], CurrentTimeTool.description)
        self.assertEqual(
            function_call_schema["responses"]["properties"]["current_time"]["type"],
            "string",
        )
