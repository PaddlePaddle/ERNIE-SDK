from __future__ import annotations

from typing import Dict, List, Type

from pydantic import Field

from erniebot_agent.memory.messages import AIMessage, HumanMessage, Message
from erniebot_agent.tools.schema import ToolParameterView

from .base import Tool


class CalculatorToolInputView(ToolParameterView):
    math_formula: str = Field(description='标准的数学公式，例如："2+3"、"3 - 4 * 6", "(3 + 4) * (6 + 4)" 等。 ')


class CalculatorToolOutputView(ToolParameterView):
    formula_result: float = Field(description="数学公式计算的结果")


class CalculatorTool(Tool):
    description: str = "CalculatorTool用于执行数学公式计算"
    input_type: Type[ToolParameterView] = CalculatorToolInputView
    ouptut_type: Type[ToolParameterView] = CalculatorToolOutputView

    async def __call__(self, math_formula: str) -> Dict[str, float]:
        # XXX: In this eval-based implementation, non-mathematical Python
        # expressions are not rejected. Should we do regex checks to ensure that
        # the input is a mathematical expression?
        try:
            code = compile(math_formula, "<string>", "eval")
        except (SyntaxError, ValueError) as e:
            raise ValueError("Invalid input expression") from e
        try:
            result = eval(code, {"__builtins__": {}}, {})
        except NameError as e:
            names_not_allowed = code.co_names
            raise ValueError(f"Names {names_not_allowed} are not allowed in the expression.") from e
        if not isinstance(result, (float, int)):
            raise ValueError("The evaluation result of the expression is not a number.")
        return {"formula_result": result}

    @property
    def examples(self) -> List[Message]:
        return [
            HumanMessage("请告诉我三加六等于多少？"),
            AIMessage(
                "",
                function_call={
                    "name": self.tool_name,
                    "thoughts": f"用户想知道3加6等于多少，我可以使用{self.tool_name}工具来计算公式，其中`math_formula`字段的内容为：'3+6'。",
                    "arguments": '{"math_formula": "3+6"}',
                },
                token_usage={
                    "prompt_tokens": 5,
                    "completion_tokens": 7,
                },  # TODO: Functional AIMessage will not add in the memory, will it add token_usage?
            ),
            HumanMessage("一加八再乘以5是多少？"),
            AIMessage(
                "",
                function_call={
                    "name": self.tool_name,
                    "thoughts": f"用户想知道1加8再乘5等于多少，我可以使用{self.tool_name}工具来计算公式，"
                    "其中`math_formula`字段的内容为：'(1+8)*5'。",
                    "arguments": '{"math_formula": "(1+8)*5"}',
                },
                token_usage={"prompt_tokens": 5, "completion_tokens": 7},  # For test only
            ),
            HumanMessage("我想知道十二除以四再加五等于多少？"),
            AIMessage(
                "",
                function_call={
                    "name": self.tool_name,
                    "thoughts": f"用户想知道12除以4再加5等于多少，我可以使用{self.tool_name}工具来计算公式，"
                    "其中`math_formula`字段的内容为：'12/4+5'。",
                    "arguments": '{"math_formula": "12/4+5"}',
                },
                token_usage={"prompt_tokens": 5, "completion_tokens": 7},  # For test only
            ),
        ]
