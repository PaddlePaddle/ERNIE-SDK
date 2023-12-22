from __future__ import annotations

import asyncio
import json
import unittest

import erniebot

from erniebot_agent.tools.calculator_tool import CalculatorTool


class TestCalculator(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = CalculatorTool()

    def run_query(self, query):
        response = erniebot.ChatCompletion.create(
            model="ernie-3.5",
            messages=[
                {
                    "role": "user",
                    "content": query,
                }
            ],
            functions=[self.tool.function_call_schema()],
            stream=False,
        )
        result = response.get_result()
        if isinstance(result, str):
            return result

        assert result["name"] == "CalculatorTool"
        arguments = json.loads(result["arguments"])
        result = asyncio.run(self.tool(**arguments))
        return result

    def test_add(self):
        result = self.run_query("3 加四等于多少")
        self.assertEqual(result["formula_result"], 7)

    def test_complex_formula(self):
        result = self.run_query("3乘以五 再加10等于多少")
        self.assertEqual(result["formula_result"], 25)
