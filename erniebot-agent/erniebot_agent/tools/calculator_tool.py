from __future__ import annotations

from typing import Type

from erniebot_agent.tools.schema import ToolView
from pydantic import Field

from .base import Tool


class CalculatorToolInputView(ToolView):
    math_formula: str = Field(description='标准的数学公式，例如："2+3"、"3 - 4 * 6", "(3 + 4) * (6 + 4)" 等。 ')


class CalculatorToolOutputView(ToolView):
    formula_result: float = Field(description="数学公式计算的结果")


class CalculatorTool(Tool):
    description: str = "CalculatorTool用于执行数学公式计算"
    inputs: Type[ToolView] = CalculatorToolInputView
    outputs: Type[ToolView] = CalculatorToolOutputView

    def __call__(self, math_formula: str) -> dict:
        return eval(math_formula)

    # @property
    # def examples(self) -> List[Message]:
    #     return [
    #         HumanMessage(),
    #         FunctionCallMessage(),
    #         HumanMessage(),
    #         FunctionCallMessage(),
    #         HumanMessage(),
    #         FunctionCallMessage(),
    #     ]
