from typing import List, Optional

from EditorAgent import EditorAgent
from RankerAgent import RankerAgent
from ResearchAgent import ResearchAgent
from ReviserAgent import ReviserAgent
from UserProxyAgent import UserProxyAgent


class ResearchTeam:
    def __init__(
        self,
        research_actor: List[ResearchAgent],
        ranker_actor: RankerAgent,
        editor_actor: EditorAgent,
        reviser_actor: ReviserAgent,
        user_agent: Optional[UserProxyAgent] = None,
    ):
        self.research_actor_instance = research_actor
        self.editor_actor_instance = editor_actor
        self.revise_actor_instance = reviser_actor
        self.ranker_actor_instance = ranker_actor
        self.user_agent = user_agent

    def _async_run(self, query, iterations=3):
        list_reports = []
        for researcher in self.research_actor_instance:
            report = researcher.run(query)
            list_reports.append(report)
        if self.user_agent is not None:
            # 让用户自己决定选取的report
            pass
        else:
            immedia_report = self.ranker_actor_instance(list_reports)

        revised_report = None
        # 可以考虑封装成一个 chain或者pipeline
        for i in range(iterations):
            if i == 0:
                markdown_report = immedia_report
            else:
                markdown_report = revised_report
            respose = self.editor_actor_instance.run(markdown_report)
            if respose["accept"] is True:
                break
            else:
                revised_report = self.revise_actor_instance.run(markdown_report, respose["msg"])

        return revised_report
