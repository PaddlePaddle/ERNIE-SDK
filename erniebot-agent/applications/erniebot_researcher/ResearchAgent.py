import json
from collections import OrderedDict
from typing import Optional

from tools.utils import add_citation, erniebot_chat, write_to_json

from erniebot_agent.agents.agent import Agent
from erniebot_agent.prompt import PromptTemplate

SUMMARIZE_MAX_LENGTH = 1800

SELECT_PROMPT = """
请你从多个综合性搜索查询{{queries}}，选取4个内容不重复搜索查询，对{{question}}问题形成客观意见。
您必须以以下格式回复一个中文字符串列表：["query 1", "query 2", "query 3", "query 4"].
"""


class ResearchAgent(Agent):
    """
    ResearchAgent, refer to
    https://github.com/assafelovic/gpt-researcher/blob/master/examples/permchain_agents/research_team.py
    """

    DEFAULT_SYSTEM_MESSAGE = """"""

    def __init__(
        self,
        name: str,
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
        aurora_db_citation,
        config=[],
        system_message: Optional[str] = None,
        use_outline=True,
        use_context_planning=True,
        save_log_path=None,
        nums_queries=4,
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
        self.aurora_db_citation = aurora_db_citation
        self.config = config
        self.save_log_path = save_log_path
        self.use_context_planning = use_context_planning
        self.nums_queries = nums_queries
        self.select_prompt = PromptTemplate(SELECT_PROMPT, input_variables=["queries", "question"])

    async def run_search_summary(self, query):
        responses = []
        url_dict = {}
        results = await self.retriever(query, top_k=3)
        length_limit = 0
        for doc in results["documents"]:
            res = await self.summarize(doc["content_se"], query)
            # Add reference to avoid hallucination
            data = {"summary": res, "url": doc["meta"]["url"], "name": doc["title"]}
            length_limit += len(res)
            if length_limit < SUMMARIZE_MAX_LENGTH:
                responses.append(data)
                key = doc["title"]
                value = doc["meta"]["url"]
                url_dict[key] = value
            else:
                print(f"summary size exceed {SUMMARIZE_MAX_LENGTH}")
                break
        # os.makedirs(os.path.dirname(f"{self.dir_path}/research-{query}.jsonl"), exist_ok=True)
        # write_to_json(f"{self.dir_path}/research-{query}.jsonl", responses)
        return responses, url_dict

    async def _run(self, query):
        """
        Runs the ResearchAgent
        Returns:
            Report
        """
        print(f"🔎 Running research for '{query}'...")
        self.config.append(("开始", f"🔎 Running research for '{query}'..."))
        self.save_log()
        # Generate Agent
        result = await self.intent_detection(query)
        self.agent, self.role = result["agent"], result["agent_role_prompt"]
        self.config.append((None, self.agent + self.role))
        self.save_log()
        if self.use_context_planning:
            sub_queries = []
            res = await self.retriever_abstract(query, top_k=3)
            context = [item["content_se"] for item in res["documents"]]
            context_content = ""
            for index, item in enumerate(context):
                sub_queries_item = await self.task_planning(
                    question=query, agent_role_prompt=self.role, context=item
                )
                sub_queries.extend(sub_queries_item)
                context_content += "第" + str(index + 1) + "篇：\n" + item + "\n"
            sub_queries_all = await self.task_planning(
                question=query, agent_role_prompt=self.role, context=context_content, is_comprehensive=True
            )
            sub_queries.extend(sub_queries_all)
            sub_queries = list(set(sub_queries))
            if len(sub_queries) > self.nums_queries:
                messages = [
                    {
                        "role": "user",
                        "content": self.select_prompt.format(queries=str(sub_queries), question=query),
                    }
                ]
                result = erniebot_chat(messages)
                start_idx = result.index("[")
                end_idx = result.rindex("]")
                result = result[start_idx : end_idx + 1]
                sub_queries = json.loads(result)
        else:
            context = ""
            # Generate Sub-Queries including original query
            sub_queries = await self.task_planning(
                question=query, agent_role_prompt=self.role, context=context
            )
        self.config.append(("任务分解", "\n".join(sub_queries)))
        self.save_log()
        # Run Sub-Queries
        meta_data = OrderedDict()
        # research_summary = ""
        paragraphs_item = []
        # summary_list=[]
        for sub_query in sub_queries:
            research_result, url_dict = await self.run_search_summary(sub_query)
            meta_data.update(url_dict)
            paragraphs_item.extend(research_result)
            self.config.append((sub_query, f"{research_result}\n\n"))
            self.save_log()
        paragraphs = []
        for item in paragraphs_item:
            if item not in paragraphs:
                paragraphs.append(item)
        research_summary = "\n\n".join([str(i) for i in paragraphs]).replace(". ", ".")
        outline = None
        # Generate Outline
        if self.use_outline:
            outline = await self.outline(sub_queries, query)
            self.config.append(("报告大纲", outline))
            self.save_log()
        else:
            outline = None
        # Conduct Research
        while True:
            try:
                report, url_index = await self.report_writing(
                    question=query,
                    research_summary=research_summary,
                    report_type=self.report_type,
                    agent_role_prompt=self.role,
                    meta_data=meta_data,
                    outline=outline,
                )
                break
            except Exception as e:
                print(e)
                self.config.append(("报错", e))
                continue
        self.config.append(("草稿", report))
        self.save_log()
        # Generate Citations
        add_citation(paragraphs, self.aurora_db_citation)
        final_report, path = await self.citation(
            report, url_index, self.agent_name, self.report_type, self.dir_path, self.aurora_db_citation
        )
        self.config.append(("草稿加引用", report))
        self.save_log()
        return final_report, path

    def save_log(self):
        write_to_json(self.save_log_path, self.config)
