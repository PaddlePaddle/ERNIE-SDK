import json
import logging
from typing import Optional

from tools.utils import ReportCallbackHandler

from erniebot_agent.agents.agent import Agent
from erniebot_agent.chat_models.erniebot import BaseERNIEBot
from erniebot_agent.memory import HumanMessage
from erniebot_agent.prompt import PromptTemplate

logger = logging.getLogger(__name__)


def get_markdown_check_prompt(report):
    prompt_markdow_str = """
    现在给你1篇报告，你需要判断报告是不是markdown格式，并给出理由。你需要输出判断理由以及判断结果，判断结果是报告是markdown形式或者报告不是markdown格式
    你的输出结果应该是个json形式，包括两个键值，一个是"判断理由"，一个是"accept"，如果你认为报告是markdown形式，则"accept"取值为True,如果你认为报告不是markdown形式，则"accept"取值为False，
    你需要判断报告是不是markdown格式，并给出理由
    {'判断理由':...,'accept':...}
    报告：{{report}}
    """
    prompt_markdow = PromptTemplate(prompt_markdow_str, input_variables=["report"])
    return prompt_markdow.format(report=report)


class RankingAgent(Agent):
    DEFAULT_SYSTEM_MESSAGE = """你是一个排序助手，你的任务就是对给定的内容和query的相关性进行排序."""

    def __init__(
        self,
        name: str,
        ranking_tool,
        llm: BaseERNIEBot,
        system_message: Optional[str] = None,
        callbacks=None,
        is_reset=False,
    ) -> None:
        self.name = name
        self.system_message = system_message or self.DEFAULT_SYSTEM_MESSAGE
        self.llm = llm

        self.ranking = ranking_tool
        self.is_reset = False
        if callbacks is None:
            self._callback_manager = ReportCallbackHandler()
        else:
            self._callback_manager = callbacks

    async def _run(self, list_reports, query):
        await self._callback_manager.on_run_start(self.name, "")
        reports = []
        for item in list_reports:
            if await self.check_format(item):
                reports.append(item)
        if len(reports) == 0:
            if self.is_reset:
                await self._callback_manager.on_run_end(self.name, "所有的report都不是markdown格式，重新生成report")
                logger.info("所有的report都不是markdown格式，重新生成report")
                return [], None
            else:
                reports = list_reports
        best_report = await self.ranking(reports, query)
        await self._callback_manager.on_run_tool(self.ranking.description, best_report)
        await self._callback_manager.on_run_end(self.name, "")
        return reports, best_report

    async def check_format(self, report):
        while True:
            try:
                messages = [HumanMessage(content=get_markdown_check_prompt(report))]
                response = await self.llm.chat(messages=messages, temperature=0.001)
                result = response.content
                l_index = result.index("{")
                r_index = result.index("}")
                result = result[l_index : r_index + 1]
                result_dict = json.loads(result)
                if result_dict["accept"] is True or result_dict["accept"] == "true":
                    return True
                elif result_dict["accept"] is False or result_dict["accept"] == "false":
                    return False
            except Exception as e:
                await self._callback_manager.on_run_error("格式检查", str(e))
                logger.error(e)
                continue
