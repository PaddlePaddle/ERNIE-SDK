from __future__ import annotations

from datetime import datetime
import erniebot
from erniebot_agent.tools.base import CalculatorTool, CurrentTimeTool, Tool

import unittest

import json


class ToolTestMixin:
    def run_query(self, query: str):
        response = erniebot.ChatCompletion.create(
            model='ernie-bot',
            messages=[{
                'role': 'user',
                'content': query,
            }],
            functions=[self.tool.function_input()],
            stream=False)

        result = response.get_result()
        if not response.is_function_response:
            return result

        assert result["name"] == self.tool.tool_name

        arguments = json.loads(result["arguments"])
        result = self.tool(**arguments)
        return result


class TestCalculator(unittest.TestCase, ToolTestMixin):
    def setUp(self) -> None:
        self.tool = CalculatorTool()

    def test_add(self):
        result = self.run_query("3 加四等于多少")
        self.assertEqual(result, 7)

    def test_multiply(self):
        result = self.run_query("3乘以五等于多少")
        self.assertEqual(result, 15)

    def test_complex_formula(self):
        result = self.run_query("3乘以五 再加10等于多少")
        self.assertEqual(result, 25)

    def test_boundary_case(self):
        result = self.run_query("左边框3个球，右边框2个球，俩框和一起，一共多少个球")
        self.assertIn("5个球", result.replace(" ", ""))


class TestCurrentTimeTool(unittest.TestCase, ToolTestMixin):
    def setUp(self) -> None:
        self.tool = CurrentTimeTool()

    def test_simple(self):
        result = self.run_query("现在北京时间什么时候？")
        now = datetime.now()

        self.assertLessEqual((now - result).microseconds, 1000)
