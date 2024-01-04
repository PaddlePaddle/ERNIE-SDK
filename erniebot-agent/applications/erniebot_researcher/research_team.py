from typing import List, Optional

from editor_actor_agent import EditorActorAgent
from ranking_agent import RankingAgent
from research_agent import ResearchAgent
from reviser_actor_agent import ReviserActorAgent
from render_agent import RenderAgent
from tools.utils import write_md_to_pdf
from user_proxy_agent import UserProxyAgent


class ResearchTeam:
    def __init__(
        self,
        research_actor: List[ResearchAgent],
        ranker_actor: RankingAgent,
        editor_actor: EditorActorAgent,
        reviser_actor: ReviserActorAgent,
        render_actor: Optional[RenderAgent] = None,
        user_agent: Optional[UserProxyAgent] = None,
        report_type: str = "research_report",
        target_path: str = "output",
    ):
        self.research_actor_instance = research_actor
        self.editor_actor_instance = editor_actor
        self.revise_actor_instance = reviser_actor
        self.ranker_actor_instance = ranker_actor
        self.render_actor_instance = render_actor
        self.user_agent = user_agent
        self.report_type = report_type
        self.target_path = target_path

    async def run(self, query, iterations=3):
        list_reports = []
        for researcher in self.research_actor_instance:
            report, _ = await researcher.run(query)
            list_reports.append(report)
        if self.user_agent is not None:
            prompt = (
                f"请你从{list_reports}个待选的多个报告草稿中，选择一个合适的报告,"
                f"直接输入序号即可，输入的序号在1和{len(self.research_actor_instance)}之间。"
            )
            index = await self.user_agent.run(prompt)
            immedia_report = list_reports[int(index) - 1]
            list_reports = [immedia_report]

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
            respose = await self.editor_actor_instance.run(markdown_report)
            if respose["accept"]:
                break
            else:
                revised_report = await self.revise_actor_instance.run(markdown_report, respose["notes"])
                # Add revise report to the list of reports
                list_reports.append(revised_report)
        if self.render_actor_instance:
            revised_report = await self.render_actor_instance.run(revised_report)
        path = write_md_to_pdf(self.report_type, self.target_path, revised_report)
        return revised_report, path
