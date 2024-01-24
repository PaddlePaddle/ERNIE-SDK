import logging
from typing import Any, List, Optional

from tools.intent_detection_tool import IntentDetectionTool
from tools.outline_generation_tool import OutlineGenerationTool
from tools.report_writing_tool import ReportWritingTool
from tools.summarization_tool import TextSummarizationTool
from tools.task_planning_tool import TaskPlanningTool
from tools.utils import JsonUtil, ReportCallbackHandler

from erniebot_agent.chat_models.erniebot import BaseERNIEBot
from erniebot_agent.memory import HumanMessage, Message, SystemMessage
from erniebot_agent.prompt import PromptTemplate

logger = logging.getLogger(__name__)
SUMMARIZE_MAX_LENGTH = 1800

SELECT_PROMPT = """
请你从多个综合性搜索查询{{queries}}，选取4个内容不重复搜索查询，对{{question}}问题形成客观意见。
您必须以以下格式回复一个中文字符串列表：["query 1", "query 2", "query 3", "query 4"].
"""

MAX_RETRY = 10


class ResearchAgent(JsonUtil):
    """
    ResearchAgent, refer to
    https://github.com/assafelovic/gpt-researcher/blob/master/examples/permchain_agents/research_team.py
    """

    DEFAULT_SYSTEM_MESSAGE = """"""

    def __init__(
        self,
        name: str,
        dir_path: str,
        report_type: str,
        retriever_abstract_db: Any,
        retriever_fulltext_db: Any,
        intent_detection_tool: IntentDetectionTool,
        task_planning_tool: TaskPlanningTool,
        report_writing_tool: ReportWritingTool,
        outline_tool: OutlineGenerationTool,
        summarize_tool: TextSummarizationTool,
        llm: BaseERNIEBot,
        system_message: Optional[SystemMessage] = None,
        use_outline: bool = True,
        use_context_planning: bool = True,
        nums_queries: int = 4,
        callbacks=None,
    ):
        self.system_message = (
            system_message.content if system_message is not None else self.DEFAULT_SYSTEM_MESSAGE
        )
        self.name = name
        self.dir_path = dir_path
        self.report_type = report_type
        self.retriever_fulltext_db = retriever_fulltext_db
        self.retriever_abstract_db = retriever_abstract_db
        self.intent_detection_tool = intent_detection_tool
        self.task_planning_tool = task_planning_tool
        self.report_writing_tool = report_writing_tool
        self.outline_tool = outline_tool
        self.summarize_tool = summarize_tool
        self.use_context_planning = use_context_planning
        self.use_outline = use_outline
        self.nums_queries = nums_queries
        self.select_prompt = PromptTemplate(SELECT_PROMPT, input_variables=["queries", "question"])
        self.llm = llm
        if callbacks is None:
            self._callback_manager = ReportCallbackHandler()
        else:
            self._callback_manager = callbacks

    async def run_search_summary(self, query: str):
        responses = []
        results = await self.retriever_fulltext_db(query, top_k=3)
        length_limit = 0
        await self._callback_manager.on_tool_start(agent=self, tool=self.summarize_tool, input_args=query)
        for doc in results["documents"]:
            res = await self.summarize_tool(doc["content"], query)
            # Add reference to avoid hallucination
            data = {"summary": res, "url": doc["meta"]["url"], "name": doc["meta"]["name"]}
            length_limit += len(res)
            if length_limit < SUMMARIZE_MAX_LENGTH:
                responses.append(data)
            else:
                logger.warning(f"summary size exceed {SUMMARIZE_MAX_LENGTH}")
                break
        await self._callback_manager.on_tool_end(self, tool=self.summarize_tool, response=responses)
        return responses

    async def run(self, query: str):
        """
        Runs the ResearchAgent
        Returns:
            Report
        """
        await self._callback_manager.on_run_start(
            agent=self, agent_name=self.name, prompt=f"🔎 Running research for '{query}'..."
        )
        # Generate Agent
        await self._callback_manager.on_tool_start(
            agent=self, tool=self.intent_detection_tool, input_args=query
        )
        result = await self.intent_detection_tool(query)
        self.agent, self.role = result["agent"], result["agent_role_prompt"]

        await self._callback_manager.on_tool_end(
            agent=self, tool=self.intent_detection_tool, response=result
        )

        if self.use_context_planning:
            sub_queries = []
            res = await self.retriever_abstract_db(query, top_k=3)
            context = [item["content"] for item in res["documents"]]
            context_content = ""
            await self._callback_manager.on_tool_start(
                agent=self, tool=self.task_planning_tool, input_args=query
            )
            for index, item in enumerate(context):
                sub_queries_item = await self.task_planning_tool(
                    question=query, agent_role_prompt=self.role, context=item
                )
                sub_queries.extend(sub_queries_item)
                context_content += "第" + str(index + 1) + "篇：\n" + item + "\n"
            sub_queries_all = await self.task_planning_tool(
                question=query, agent_role_prompt=self.role, context=context_content, is_comprehensive=True
            )
            sub_queries.extend(sub_queries_all)
            sub_queries = list(set(sub_queries))
            # Sampling 4 sub-queries
            if len(sub_queries) > self.nums_queries:
                messages: List[Message] = [
                    HumanMessage(content=self.select_prompt.format(queries=str(sub_queries), question=query))
                ]
                responese = await self.llm.chat(messages)
                result = responese.content
                sub_queries = self.parse_json(result, "[", "]")
            await self._callback_manager.on_tool_end(
                self, tool=self.task_planning_tool, response=sub_queries
            )
        else:
            await self._callback_manager.on_tool_start(
                agent=self, tool=self.task_planning_tool, input_args=query
            )
            # Generate Sub-Queries including original query
            sub_queries = await self.task_planning_tool(question=query, agent_role_prompt=self.role)
            await self._callback_manager.on_tool_end(
                self, tool=self.task_planning_tool, response=sub_queries
            )
        # Run Sub-Queries
        paragraphs_item = []
        for sub_query in sub_queries:
            research_result = await self.run_search_summary(sub_query)
            paragraphs_item.extend(research_result)
        paragraphs = []
        for item in paragraphs_item:
            if item not in paragraphs:
                paragraphs.append(item)
        # 1. 摘要 ==> 1.摘要 for avoiding erniebot request error
        research_summary = "\n\n".join([str(i) for i in paragraphs]).replace(". ", ".")

        await self._callback_manager.on_tool_start(
            agent=self, tool=self.outline_tool, input_args=sub_queries
        )
        # Generate Outline
        outline = None
        if self.use_outline:
            outline = await self.outline_tool(sub_queries, query)

        await self._callback_manager.on_tool_end(self, tool=self.outline_tool, response=outline)

        await self._callback_manager.on_tool_start(
            agent=self, tool=self.report_writing_tool, input_args=query
        )
        # Conduct Research
        retry_count = 0
        while True:
            try:
                report, path = await self.report_writing_tool(
                    question=query,
                    research_summary=research_summary,
                    report_type=self.report_type,
                    agent_role_prompt=self.role,
                    outline=outline,
                    agent_name=self.name,
                    dir_path=self.dir_path,
                )
                break
            except Exception as e:
                await self._callback_manager.on_tool_error(self, tool=self.report_writing_tool, error=e)
                retry_count += 1
                if retry_count > MAX_RETRY:
                    raise Exception(f"Failed to conduct research for {query} after {MAX_RETRY} times.")
                continue
        await self._callback_manager.on_tool_end(
            self, tool=self.report_writing_tool, response={"report": report, "file_path": path}
        )
        await self._callback_manager.on_run_end(agent=self, agent_name=self.name, response=f"报告存储在{path}")
        return report, paragraphs
