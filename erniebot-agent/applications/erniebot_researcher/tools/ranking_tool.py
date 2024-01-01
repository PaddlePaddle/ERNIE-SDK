from __future__ import annotations

import json
from typing import List, Type

from pydantic import Field

from erniebot_agent.tools.base import Tool
from erniebot_agent.tools.schema import ToolParameterView

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
        if len(reports) == 1:
            return reports[0]
        elif len(reports) > 1:
            socres_all = []
            for item in reports:
                content = rank_report_prompt(report=item, query=query)
                messages = [{"role": "user", "content": content}]
                while True:
                    try:
                        if len(content) <= 4800:
                            result = erniebot_chat(messages=messages, temperature=1e-10)
                        else:
                            result = erniebot_chat(
                                messages=messages, temperature=1e-10, model="ernie-bot-8k"
                            )
                        l_index = result.index("{")
                        r_index = result.rindex("}")
                        result = result[l_index : r_index + 1]
                        result_dict = json.loads(result)
                        socre = int(result_dict["报告总得分"])
                        socres_all.append(socre)
                        break
                    except Exception as e:
                        print(e)
                continue
            best_index = socres_all.index(max(socres_all))
            rank_result = reports[best_index]
            return rank_result
