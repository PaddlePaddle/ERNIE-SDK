from .base import Tool


class CalculatorTool(Tool):
    """"""

    def __call__(self, math_formula: str) -> float:
        """CalculatorTool用于执行数学公式计算

        Args:
            math_formula (str): 标准的数学公式，例如："2+3"、"3 - 4 * 6", "(3 + 4) * (6 + 4)" 等。

        Examples:
            >>> {"role": "user", "content": "请告诉我三加六等于多少"}
            >>> {"role": "assistant", "content": null, "function_call": {"name": "CalculatorTool", "arguments": "{'math_formula':'3+6'}"}}
            >>> {"role": "user", "content": "一加八再乘以五是多少"}
            >>> {"role": "assistant", "content": null, "function_call": {"name": "CalculatorTool", "arguments": "{'math_formula':'(1+8)*5'}"}}
            >>> {"role": "user", "content": "我想知道十二除以 4 再加 5 等于多少"}
            >>> {"role": "assistant", "content": null, "function_call": {"name": "CalculatorTool", "arguments": "{'math_formula':'12/4+5'}"}}

        Returns:
            int: 数学公式计算的结果
        """  # noqa: E501

        return eval(math_formula)
