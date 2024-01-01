import json
from typing import Optional

from tools.prompt_utils import prompt_markdow
from tools.utils import erniebot_chat, write_to_json

from erniebot_agent.agents.base import Agent


class RankingAgent(Agent):
    DEFAULT_SYSTEM_MESSAGE = """你是一个排序助手，你的任务就是对给定的内容和query的相关性进行排序."""

    def __init__(
        self,
        name: str,
        ranking_tool,
        system_message: Optional[str] = None,
        config: list = [],
        save_log_path=None,
        is_reset=False,
    ) -> None:
        self.name = name
        self.system_message = system_message or self.DEFAULT_SYSTEM_MESSAGE  # type: ignore

        self.ranking = ranking_tool
        self.config = config
        self.save_log_path = save_log_path
        self.is_reset = False

    async def _async_run(self, list_reports, query):
        reports = []
        for item in list_reports:
            if self.check_format(item):
                reports.append(item)
        if len(reports) == 0:
            if self.is_reset:
                print("所有的report都不是markdown格式，重新生成report")
                return [], None
            else:
                reports = list_reports
        best_report = await self.ranking(reports, query)
        self.config.append(("最好的report", best_report))
        if self.save_log_path:
            self.save_log()
        return reports, best_report

    def save_log(self):
        write_to_json(self.save_log_path, self.config, mode="a")

    def check_format(self, report):
        while True:
            try:
                messages = [{"role": "user", "content": prompt_markdow.format(report=report)}]
                result = erniebot_chat(messages=messages, temperature=0.001)
                l_index = result.index("{")
                r_index = result.index("}")
                result = result[l_index : r_index + 1]
                result_dict = json.loads(result)
                if result_dict["accept"] is True or result_dict["accept"] == "true":
                    return True
                elif result_dict["accept"] is False or result_dict["accept"] == "false":
                    return False
            except Exception as e:
                print(e)
                continue
