from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Type

from erniebot_agent.message import AIMessage, HumanMessage, Message
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
            AIMessage(None, function_call={"current_time": "2023年11月8号 12点:08分:36秒"}),
            HumanMessage("现在是什么时候？"),
            AIMessage(None, function_call={"current_time": "2023年11月8号 12点:08分:36秒"}),
            HumanMessage("请问现在是什么时间"),
            AIMessage(None, function_call={"current_time": "2023年11月8号 12点:08分:36秒"}),
        ]
