from __future__ import annotations

import json
import logging
from typing import List, Type

from pydantic import Field

from erniebot_agent.chat_models.erniebot import BaseERNIEBot
from erniebot_agent.memory import HumanMessage
from erniebot_agent.prompt import PromptTemplate
from erniebot_agent.tools.base import Tool
from erniebot_agent.tools.schema import ToolParameterView

logger = logging.getLogger(__name__)


def rank_report_prompt(report, query):
    prompt = """现在给你1篇报告，现在你需要严格按照以下的标准，对这个报告进行打分，越符合标准得分越高，打分区间在0-10之间，
    你输出的应该是一个json格式，json中的键值为"打分理由"和"报告总得分"，{'打分理由':...,'报告总得分':...}
    对报告进行打分,打分标准如下：
    1.仔细检查报告格式，报告必须是完整的，包括标题、摘要、正文、参考文献等，完整性越高，得分越高，这一点最高给4分。
    3.仔细检查报告内容，报告内容与{{query}}问题相关性越高得分越高，这一点最高给4分。
    4.仔细检查报告格式，标题是否有"#"符号标注，这一点最高给2分，没有"#"给0分，有"#"给1分。
    5.仔细检查报告格式，报告的标题句结尾不能有任何中文符号，标题结尾有中文符号给0分，标题结尾没有中文符号给1分。
    以下是这篇报告的内容：{{content}}
    请你记住，你需要根据打分标准给出每篇报告的打分理由，打分理由报告
    最后给出打分结果和最终的打分列表。
    你的输出需要按照以下格式进行输出：
    为了对这报告进行打分，我将根据给定的标准进行评估。报告的打分理由将基于以下五个标准：
    1) 是否包含标题、摘要、正文、参考文献等，3) 内容与问题的相关性，4) 标题是否有"#"标注，5) 标题是否有中文符号。
    """
    prompt_socre = PromptTemplate(prompt, input_variables=["query", "content"])
    strs = prompt_socre.format(content=report, query=query)
    return strs


class TextRankingToolInputView(ToolParameterView):
    query: str = Field(description="Chunk of text to ranking")


class TextRankingToolOutputView(ToolParameterView):
    document: str = Field(description="content")


class TextRankingTool(Tool):
    description: str = "text ranking tool"
    input_type: Type[ToolParameterView] = TextRankingToolInputView
    ouptut_type: Type[ToolParameterView] = TextRankingToolOutputView

    def __init__(self, llm: BaseERNIEBot, llm_long: BaseERNIEBot) -> None:
        super().__init__()
        self.llm = llm
        self.llm_long = llm_long

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
                messages = [HumanMessage(content)]
                while True:
                    try:
                        if len(content) <= 4800:
                            response = await self.llm.chat(messages=messages, temperature=1e-10)
                        else:
                            response = await self.llm_long.chat(
                                messages=messages,
                                temperature=1e-10,
                            )
                        result = response.text
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
