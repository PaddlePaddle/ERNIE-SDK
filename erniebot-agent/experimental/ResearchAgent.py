from typing import Any, List, Optional

from erniebot_agent.agents.base import Agent


class ResearchAgent(Agent):
    """
    ResearchAgent, refer to
    https://github.com/assafelovic/gpt-researcher/blob/master/examples/permchain_agents/research_team.py
    """

    DEFAULT_SYSTEM_MESSAGE = """"""

    def __init__(
        self,
        name: str,
        query,
        report_type,
        retriever_tool,
        intent_detection_tool,
        task_planning_tool,
        report_writing_tool,
        outline_tool,
        citation_tool,
        config=None,
        system_message: Optional[str] = None,
    ):
        """
        Initialize the ResearchAgent class.
        Args:
            query:
            report_type:
            ......
        """
        self.name = name
        self.system_message = system_message or self.DEFAULT_SYSTEM_MESSAGE  # type: ignore

        self.query = query
        self.agent = None
        self.role = None
        self.report_type = report_type
        self.cfg = config
        self.retriever = retriever_tool
        self.intent_detection = intent_detection_tool
        self.task_planning = task_planning_tool
        self.report_writing = report_writing_tool
        self.outline = outline_tool
        self.citation = citation_tool
        self.context: List[Any] = []

    async def _async_run(self):
        """
        Runs the ResearchAgent
        Returns:
            Report
        """
        print(f"🔎 Running research for '{self.query}'...")
        # Generate Agent
        self.agent, self.role = await self.intent_detection(self.query, self.cfg)
        # Generate Sub-Queries including original query
        sub_queries = await self.task_planning(self.query, self.role, self.cfg) + [self.query]
        # Run Sub-Queries
        for sub_query in sub_queries:
            context = await self.retriever(sub_query)
            summarize_text = await self.summarize(sub_query, context)
            self.context.append(summarize_text)

        # Generate Outline
        outline = await self.outline(self.context, self.query)
        # Conduct Research
        report = await self.report_writing(
            query=self.query,
            context=self.context,
            outline=outline,
            agent_role_prompt=self.role,
            report_type=self.report_type,
            cfg=self.cfg,
        )
        # Generate Citations
        report = self.citation(report)
        return report
