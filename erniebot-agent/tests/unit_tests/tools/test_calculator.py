import unittest

from erniebot_agent.tools.calculator_tool import CalculatorTool


class TestCalculatorTool(unittest.TestCase):
    def test_schema(self):
        calculator = CalculatorTool()
        function_call_schema = calculator.function_call_schema()

        self.assertEqual(function_call_schema["description"], "CalculatorTool用于执行数学公式计算")
        self.assertIn("math_formula", function_call_schema["parameters"]["properties"])
        self.assertEqual(function_call_schema["parameters"]["properties"]["math_formula"]["type"], "string")

        self.assertEqual(function_call_schema["parameters"]["properties"]["math_formula"]["type"], "string")

        self.assertEqual(
            function_call_schema["responses"]["properties"]["formula_result"]["type"],
            "number",
        )
