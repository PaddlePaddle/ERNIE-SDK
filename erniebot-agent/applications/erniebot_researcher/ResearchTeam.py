from typing import List, Optional

from EditorActorAgent import EditorActorAgent
from RankingAgent import RankingAgent
from ResearchAgent import ResearchAgent
from ReviserActorAgent import ReviserActorAgent
from UserProxyAgent import UserProxyAgent


class ResearchTeam:
    def __init__(
        self,
        research_actor: List[ResearchAgent],
        ranker_actor: RankingAgent,
        editor_actor: EditorActorAgent,
        reviser_actor: ReviserActorAgent,
        user_agent: Optional[UserProxyAgent] = None,
    ):
        self.research_actor_instance = research_actor
        self.editor_actor_instance = editor_actor
        self.revise_actor_instance = reviser_actor
        self.ranker_actor_instance = ranker_actor
        self.user_agent = user_agent

    async def _async_run(self, query, iterations=3):
        list_reports = []
        for researcher in self.research_actor_instance:
            report, _ = await researcher._run(query)
            list_reports.append(report)
        if self.user_agent is not None:
            prompt = (
                f"请你从{list_reports}个待选的多个报告草稿中，选择一个合适的报告,"
                "直接输入序号即可，输入的序号在1和"
                str(len(self.research_actor_instance))
                "之间。"
            )
            index = await self.user_agent._run(prompt)
            immedia_report = list_reports[int(index) - 1]
        else:
            immedia_report = await self.ranker_actor_instance._run(list_reports, query)
        revised_report = immedia_report
        for i in range(iterations):
            if i == 0:
                markdown_report = immedia_report
            else:
                markdown_report = revised_report
            respose = await self.editor_actor_instance._run(markdown_report)
            if respose["accept"]:
                break
            else:
                revised_report = await self.revise_actor_instance._run(markdown_report, respose["notes"])
        return revised_report
