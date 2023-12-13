import json
from typing import Optional

from erniebot_agent.agents.base import Agent
from tools.utils import erniebot_chat


class RankingAgent(Agent):
    DEFAULT_SYSTEM_MESSAGE = """你是一个排序助手，你的任务就是对给定的内容和query的相关性进行排序."""

    def __init__(
        self,
        name: str,
        summarize_tool,
        ranking_tool,
        system_message: Optional[str] = None,
        use_summarize: bool = False,
    ) -> None:
        self.name = name
        self.system_message = system_message or self.DEFAULT_SYSTEM_MESSAGE  # type: ignore

        self.summarize = summarize_tool
        self.ranking = ranking_tool
        self.use_summarize = use_summarize

    async def _async_run(self, list_reports, query):
        # filter one
        new_list_reports = []
        for report in list_reports:
            response = self.check_format(report)
            if response["accept"] == "true":
                new_list_reports.append(report)
        if self.use_summarize:
            # summarize
            summarize_list = []
            for item in new_list_reports:
                summarize_text = await self.summarize(item, query)
                summarize_list.append(summarize_text)
            _, index = await self.ranking(summarize_list, query)
            best_report = new_list_reports[index - 1]
        else:
            best_report, _ = await self.ranking(new_list_reports, query)
        # Select the best one (first)
        return best_report

    def check_format(self, report):
        prompt = """你是一名校对员。
            你应该校对以下事项：
            - 这份草稿必须充分回答原始问题。
            - 这份草稿必须按照APA格式编写。
            - 这份草稿必须不包含低级的句法错误。
            校对结果，请以json的格式输出：
            如果这份草搞满足上述所有事项，则输出：{"accept":"true"} 否则输出： {"accept": "false"}
            """
        messages = [{"role": "user", "content": "草稿为:\n\n" + report}]
        respose = erniebot_chat(messages=messages, system=prompt)
        start = respose.index("{")
        end = respose.rindex("}")
        respose = respose[start : end + 1]
        return json.loads(respose)
