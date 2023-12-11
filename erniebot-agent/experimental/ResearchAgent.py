import os
from collections import OrderedDict
from typing import Optional

from erniebot_agent.agents.base import Agent
from erniebot_agent.tools.utils import write_to_json

SUMMARIZE_MAX_LENGTH = 1800


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
        agent_name,
        dir_path,
        report_type,
        retriever_abstract_tool,
        retriever_tool,
        intent_detection_tool,
        task_planning_tool,
        report_writing_tool,
        outline_tool,
        citation_tool,
        summarize_tool,
        config=None,
        system_message: Optional[str] = None,
        use_outline=True,
        use_context_planning=True,
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
        self.dir_path = dir_path
        self.report_type = report_type
        self.cfg = config
        self.retriever = retriever_tool
        self.retriever_abstract = retriever_abstract_tool
        self.intent_detection = intent_detection_tool
        self.task_planning = task_planning_tool
        self.report_writing = report_writing_tool
        self.outline = outline_tool
        self.citation = citation_tool
        self.summarize = summarize_tool
        self.use_context_planning = use_context_planning
        self.use_outline = use_outline
        self.agent_name = agent_name

    async def run_search_summary(self, query):
        responses = []
        url_dict = {}
        results = await self.retriever(query, top_k=3)
        length_limit = 0
        for doc in results:
            res = await self.summarize(doc.page_content, query)
            # Add reference to avoid hallucination
            data = {"summary": res, "url": doc.metadata["url"], "name": doc.metadata["name"]}
            length_limit += len(res)
            if length_limit < SUMMARIZE_MAX_LENGTH:
                responses.append(data)
                key = doc.metadata["name"]
                value = doc.metadata["url"]
                url_dict[key] = value
            else:
                print(f"summary size exceed {SUMMARIZE_MAX_LENGTH}")
                break
        os.makedirs(os.path.dirname(f"{self.dir_path}/research-{query}.jsonl"), exist_ok=True)
        write_to_json(f"{self.dir_path}/research-{query}.jsonl", responses)
        return responses, url_dict

    async def _async_run(self):
        """
        Runs the ResearchAgent
        Returns:
            Report
        """
        print(f"🔎 Running research for '{self.query}'...")
        # Generate Agent
        result = await self.intent_detection(self.query)
        self.agent, self.role = result["agent"], result["agent_role_prompt"]
        use_context_planning = True
        if use_context_planning:
            res = await self.retriever_abstract(self.query)
            context = [item.page_content for item in res]
            context = "\n".join(context)
        else:
            context = []

        # Generate Sub-Queries including original query
        sub_queries = await self.task_planning(
            question=self.query, agent_role_prompt=self.role, context=context
        )
        # Run Sub-Queries
        meta_data = OrderedDict()
        research_summary = ""
        paragraphs = []
        for sub_query in sub_queries:
            research_result, url_dict = await self.run_search_summary(sub_query)
            research_summary += f"{research_result}\n\n"
            meta_data.update(url_dict)
            paragraphs.extend(research_result)
        outline = None

        # Generate Outline
        if self.use_outline:
            outline = await self.outline(sub_queries, self.query)
        else:
            outline = None
        # Conduct Research
        breakpoint()
        report, url_index = await self.report_writing(
            question=self.query,
            research_summary=research_summary,
            report_type=self.report_type,
            agent_role_prompt=self.role,
            meta_data=meta_data,
            outline=outline,
        )
        # Generate Citations
        final_report, path = await self.citation(
            report, paragraphs, url_index, self.agent_name, self.report_type, self.dir_path
        )
        return final_report, path
