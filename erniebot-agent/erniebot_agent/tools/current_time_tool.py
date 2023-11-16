from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Type

from erniebot_agent.messages import AIMessage, HumanMessage, Message
from erniebot_agent.tools.schema import ToolParameterView
from pydantic import Field

from .base import Tool


class CurrentTimeToolOutputView(ToolParameterView):
    current_time: str = Field(description="当前时间")


class CurrentTimeTool(Tool):
    description: str = "CurrentTimeTool 用于获取当前时间"
    ouptut_type: Type[ToolParameterView] = CurrentTimeToolOutputView

    async def __call__(self) -> Dict[str, str]:
        return {"current_time": datetime.strftime(datetime.now(), "%Y年%m月%d号 %点:%分:%秒")}

    @property
    def examples(self) -> List[Message]:
        return [
            HumanMessage("现在几点钟了"),
            AIMessage(
                "",
                function_call={
                    "name": self.tool_name,
                    "thoughts": f"用户想知道现在几点了，我可以使用{self.tool_name}来获取当前时间，并从其中获得当前小时时间。",
                    "arguments": "{}",
                },
            ),
            HumanMessage("现在是什么时候？"),
            AIMessage(
                "",
                function_call={
                    "name": self.tool_name,
                    "thoughts": f"用户想知道现在几点了，我可以使用{self.tool_name}来获取当前时间",
                    "arguments": "{}",
                },
            ),
        ]
