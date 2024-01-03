from __future__ import annotations

import json
import logging
from typing import List, Type

from pydantic import Field

from erniebot_agent.tools.base import Tool
from erniebot_agent.tools.schema import ToolParameterView

from .prompt_utils import rank_report_prompt
from .utils import erniebot_chat

logger = logging.getLogger(__name__)


class TextRankingToolInputView(ToolParameterView):
    query: str = Field(description="Chunk of text to ranking")


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
            scores_all = []
            for item in reports:
                content = rank_report_prompt(report=item, query=query)
                messages = [{"role": "user", "content": content}]
                while True:
                    try:
                        if len(content) <= 4800:
                            result = erniebot_chat(messages=messages, temperature=1e-10)
                        else:
                            result = erniebot_chat(
                                messages=messages, temperature=1e-10, model="ernie-longtext"
                            )
                        l_index = result.index("{")
                        r_index = result.rindex("}")
                        result = result[l_index : r_index + 1]
                        result_dict = json.loads(result)
                        socre = int(result_dict["报告总得分"])
                        scores_all.append(socre)
                        break
                    except Exception as e:
                        logger.error(e)
                continue
            best_index = scores_all.index(max(scores_all))
            rank_result = reports[best_index]
            return rank_result
