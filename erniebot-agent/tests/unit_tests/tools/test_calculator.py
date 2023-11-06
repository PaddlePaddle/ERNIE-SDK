import unittest

from erniebot_agent.tools.calculator_tool import CalculatorTool


class TestCalculatorTool(unittest.TestCase):
    def test_call(self):
        calculator = CalculatorTool()
        self.assertEqual(calculator("2+3"), 5)
        self.assertEqual(calculator("3 - 4 * 6"), -21)
        self.assertEqual(calculator("(3 + 4) * (6 + 4)"), 70)

    def test_schema(self):
        calculator = CalculatorTool()
        function_call_schema = calculator.function_call_schema()
        self.assertEqual(function_call_schema["description"], "CalculatorTool用于执行数学公式计算")
        self.assertIn("math_formula", function_call_schema["parameters"]["properties"])
        self.assertEqual(function_call_schema["parameters"]["properties"]["math_formula"]["type"], "string")

        self.assertEqual(function_call_schema["parameters"]["properties"]["math_formula"]["type"], "string")

        self.assertEqual(
            function_call_schema["responses"]["properties"]["return_value_of_CalculatorTool"]["type"],
            "integer",
        )
