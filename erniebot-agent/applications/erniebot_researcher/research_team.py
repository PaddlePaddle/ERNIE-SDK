import asyncio
from typing import List, Optional

from editor_actor_agent import EditorActorAgent
from fact_check_agent import FactCheckerAgent
from polish_agent import PolishAgent
from ranking_agent import RankingAgent
from research_agent import ResearchAgent
from reviser_actor_agent import ReviserActorAgent
from user_proxy_agent import UserProxyAgent


class ResearchTeam:
    def __init__(
        self,
        research_actor: List[ResearchAgent],
        ranker_actor: RankingAgent,
        editor_actor: EditorActorAgent,
        reviser_actor: ReviserActorAgent,
        checker_actor: FactCheckerAgent,
        polish_actor: Optional[PolishAgent] = None,
        user_agent: Optional[UserProxyAgent] = None,
        use_reflection: bool = False,
    ):
        self.research_actor_instance = research_actor
        self.editor_actor_instance = editor_actor
        self.revise_actor_instance = reviser_actor
        self.ranker_actor_instance = ranker_actor
        self.polish_actor_instance = polish_actor
        self.checker_actor_instance = checker_actor
        self.user_agent = user_agent
        self.polish_actor = polish_actor
        self.use_reflection = use_reflection

    async def run(self, query, iterations=3):
        tasks_researchers = [researcher.run(query) for researcher in self.research_actor_instance]
        result_researchers = await asyncio.gather(*tasks_researchers)
        list_reports = [{"report": result[0], "paragraphs": result[1]} for result in result_researchers]
        if self.user_agent is not None:
            prompt = (
                f"请你从{list_reports}个待选的多个报告草稿中，选择一个合适的报告,"
                f"直接输入序号即可，输入的序号在1和{len(self.research_actor_instance)}之间。"
            )
            index = await self.user_agent.run(prompt)
            immedia_report = list_reports[int(index) - 1]
            list_reports = [immedia_report]

        if self.use_reflection:
            for i in range(iterations):
                # Listwise ranking
                if len(list_reports) > 1:
                    # Filter out low quality report and ranking the remaining reports
                    list_reports, immedia_report = await self.ranker_actor_instance.run(list_reports, query)
                    if len(list_reports) == 0:
                        raise Exception("Current report is not good to optimize.")
                elif len(list_reports) == 1:
                    immedia_report = list_reports[0]
                else:
                    raise Exception("No report to optimize.")
                revised_report = immedia_report
                if i == 0:
                    markdown_report = immedia_report
                else:
                    markdown_report = revised_report
                # report, (meta_data, paragraphs)
                response = await self.editor_actor_instance.run(markdown_report)
                if response["accept"]:
                    break
                else:
                    revised_report = await self.revise_actor_instance.run(markdown_report, response["notes"])
                    # Add revise report to the list of reports
                    list_reports.append(revised_report)
        else:
            if len(list_reports) > 1:
                # Filter out low quality report and ranking the remaining reports
                list_reports, immedia_report = await self.ranker_actor_instance.run(list_reports, query)
                if len(list_reports) == 0:
                    raise Exception("Current report is not good to optimize.")
            elif len(list_reports) == 1:
                immedia_report = list_reports[0]

            revised_report = immedia_report
        checked_report = await self.checker_actor_instance.run(report=revised_report["report"])
        revised_report, path = await self.polish_actor_instance.run(
            report=checked_report,
            summarize=revised_report["paragraphs"],
        )
        return revised_report, path
