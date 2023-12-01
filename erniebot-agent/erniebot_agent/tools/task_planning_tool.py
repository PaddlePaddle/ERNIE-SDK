from __future__ import annotations

from typing import Dict, List, Type

from erniebot_agent.tools.schema import ToolParameterView
from pydantic import Field

from .base import Tool


class TaskPlanningToolInputView(ToolParameterView):
    query: str = Field(description="Chunk of text to summarize")


class TaskPlanningToolOutputView(ToolParameterView):
    document: str = Field(description="content")


class TaskPlanningTool(Tool):
    description: str = "text summarization tool"
    input_type: Type[ToolParameterView] = TaskPlanningToolInputView
    ouptut_type: Type[ToolParameterView] = TaskPlanningToolOutputView

    async def __call__(
        self,
        content: str,
    ) -> Dict[str, List[str]]:
        # map reduce

        raise NotImplementedError
