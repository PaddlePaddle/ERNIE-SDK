from typing import Optional

from erniebot_agent.agents.base import Agent


class RankingAgent(Agent):
    DEFAULT_SYSTEM_MESSAGE = """你是一个排序助手，你的任务就是对给定的内容和query的相关性进行排序."""

    def __init__(
        self, name: str, summarize_tool, ranking_tool, system_message: Optional[str] = None
    ) -> None:
        self.name = name
        self.system_message = system_message or self.DEFAULT_SYSTEM_MESSAGE  # type: ignore

        self.summarize = summarize_tool
        self.ranking = ranking_tool

    def _async_run(self, list_reports):
        # filter one
        list_reports = []
        for report in list_reports:
            response = self.check_format(report)
            if response["accept"] is True:
                list_reports.append(report)
        # summarize
        summarize_list = []
        for item in list_reports:
            summarize_text = self.summarize(item)
            summarize_list.append(summarize_text)

        sorted_idx = self.ranking(summarize_list)
        # Select the best one (first)
        return list_reports[sorted_idx[0]]

    def check_format(self, report):
        pass
