from __future__ import annotations

from typing import Dict, List, Type

from erniebot_agent.tools.schema import ToolParameterView
from pydantic import Field

from .base import Tool


class TextRankingToolInputView(ToolParameterView):
    query: str = Field(description="Chunk of text to summarize")


class TextRankingToolOutputView(ToolParameterView):
    document: str = Field(description="content")


class TextRankingTool(Tool):
    description: str = "text ranking tool"
    input_type: Type[ToolParameterView] = TextRankingToolInputView
    ouptut_type: Type[ToolParameterView] = TextRankingToolOutputView

    async def __call__(
        self,
        content: str,
    ) -> Dict[str, List[str]]:
        # map reduce
        raise NotImplementedError
