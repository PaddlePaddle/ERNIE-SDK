from __future__ import annotations

from typing import Dict, List, Type

from erniebot_agent.tools.schema import ToolParameterView
from pydantic import Field

from .base import Tool


class TextSummarizationToolInputView(ToolParameterView):
    query: str = Field(description="Chunk of text to summarize")


class TextSummarizationToolOutputView(ToolParameterView):
    document: str = Field(description="content")


class TextSummarizationTool(Tool):
    description: str = "text summarization tool"
    input_type: Type[ToolParameterView] = TextSummarizationToolInputView
    ouptut_type: Type[ToolParameterView] = TextSummarizationToolOutputView

    async def __call__(
        self,
        content: str,
    ) -> Dict[str, List[str]]:
        # map reduce
        raise NotImplementedError
