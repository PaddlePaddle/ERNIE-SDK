from __future__ import annotations

from typing import List, Type

from erniebot_agent.tools.base import Tool
from erniebot_agent.tools.schema import ToolParameterView
from pydantic import Field

from .prompt_utils import rank_report_prompt
from .utils import erniebot_chat


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
        reports: List,
        query: str,
    ):
        messages = [{"role": "user", "content": rank_report_prompt(reports=reports, query=query)}]
        rank_result = erniebot_chat(messages, model="ernie-bot-8k")
        rank_list = rank_result.split(">")
        for item in rank_list:
            report_num = item.strip()[1:-1]
            if int(report_num) <= len(reports):
                break
        final_report = reports[int(report_num) - 1]
        return final_report
