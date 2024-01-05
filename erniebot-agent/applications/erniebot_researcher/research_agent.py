import json
import logging
from collections import OrderedDict
from typing import Optional

from tools.utils import ReportCallbackHandler

from erniebot_agent.agents.callback.callback_manager import CallbackManager
from erniebot_agent.chat_models.erniebot import BaseERNIEBot
from erniebot_agent.memory import HumanMessage, SystemMessage
from erniebot_agent.prompt import PromptTemplate

logger = logging.getLogger(__name__)
SUMMARIZE_MAX_LENGTH = 1800

SELECT_PROMPT = """
请你从多个综合性搜索查询{{queries}}，选取4个内容不重复搜索查询，对{{question}}问题形成客观意见。
您必须以以下格式回复一个中文字符串列表：["query 1", "query 2", "query 3", "query 4"].
"""

MAX_RETRY = 10


class ResearchAgent:
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
        summarize_tool,
        llm: BaseERNIEBot,
        system_message: Optional[SystemMessage] = None,
        use_outline=True,
        use_context_planning=True,
        nums_queries=4,
        callbacks=None,
    ):
        """
        Initialize the ResearchAgent class.
        Args:
            query:
            report_type:
            ......
        """

        self.name = name
        self.system_message = (
            system_message.content if system_message is not None else self.DEFAULT_SYSTEM_MESSAGE
        )
        self.dir_path = dir_path
        self.report_type = report_type
        self.retriever = retriever_tool
        self.retriever_abstract = retriever_abstract_tool
        self.intent_detection = intent_detection_tool
        self.task_planning = task_planning_tool
        self.report_writing = report_writing_tool
        self.outline = outline_tool
        self.summarize = summarize_tool
        self.use_context_planning = use_context_planning
        self.use_outline = use_outline
        self.agent_name = agent_name
        self.use_context_planning = use_context_planning
        self.nums_queries = nums_queries
        self.select_prompt = PromptTemplate(SELECT_PROMPT, input_variables=["queries", "question"])
        self.llm = llm
        if callbacks is None:
            self._callback_manager = CallbackManager([ReportCallbackHandler()])
        else:
            self._callback_manager = callbacks

    async def run_search_summary(self, query):
        responses = []
        url_dict = {}
        results = self.retriever.search(query, top_k=3)
        length_limit = 0
        await self._callback_manager.on_tool_start(agent=self, tool=self.summarize, input_args=query)
        for doc in results:
            res = await self.summarize(doc["content"], query)
            # Add reference to avoid hallucination
            data = {"summary": res, "url": doc["url"], "name": doc["title"]}
            length_limit += len(res)
            if length_limit < SUMMARIZE_MAX_LENGTH:
                responses.append(data)
                key = doc["title"]
                value = doc["url"]
                url_dict[key] = value
            else:
                logger.warning(f"summary size exceed {SUMMARIZE_MAX_LENGTH}")
                break
        await self._callback_manager.on_tool_end(self, tool=self.summarize, response=responses)
        return responses, url_dict

    async def run(self, query):
        """
        Runs the ResearchAgent
        Returns:
            Report
        """
        await self._callback_manager.on_run_start(
            agent=self, agent_name=self.name, prompt=f"🔎 Running research for '{query}'..."
        )
        # Generate Agent
        await self._callback_manager.on_tool_start(agent=self, tool=self.intent_detection, input_args=query)
        result = await self.intent_detection(query)
        self.agent, self.role = result["agent"], result["agent_role_prompt"]

        await self._callback_manager.on_tool_end(agent=self, tool=self.intent_detection, response=result)

        await self._callback_manager.on_tool_start(agent=self, tool=self.task_planning, input_args=query)
        if self.use_context_planning:
            sub_queries = []
            res = self.retriever_abstract.search(query, top_k=3)
            context = [item["content"] for item in res]
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
            # Sampling 4 sub-queries
            if len(sub_queries) > self.nums_queries:
                messages = [
                    HumanMessage(content=self.select_prompt.format(queries=str(sub_queries), question=query))
                ]
                responese = await self.llm.chat(messages)
                result = responese.content
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
        await self._callback_manager.on_tool_end(self, tool=self.task_planning, response=sub_queries)
        # Run Sub-Queries
        meta_data = OrderedDict()
        paragraphs_item = []
        for sub_query in sub_queries:
            research_result, url_dict = await self.run_search_summary(sub_query)
            meta_data.update(url_dict)
            paragraphs_item.extend(research_result)

        paragraphs = []
        for item in paragraphs_item:
            if item not in paragraphs:
                paragraphs.append(item)
        # 1. 摘要 ==> 1.摘要 for avoiding erniebot request error
        research_summary = "\n\n".join([str(i) for i in paragraphs]).replace(". ", ".")
        outline = None

        await self._callback_manager.on_tool_start(agent=self, tool=self.outline, input_args=sub_queries)
        # Generate Outline
        if self.use_outline:
            outline = await self.outline(sub_queries, query)
            await self._callback_manager.on_run_tool(tool_name=self.outline.description, response=outline)
        else:
            outline = None
        await self._callback_manager.on_tool_end(self, tool=self.outline, response=outline)

        await self._callback_manager.on_tool_start(agent=self, tool=self.report_writing, input_args=query)
        # Conduct Research
        retry_count = 0
        while True:
            try:
                report, path = await self.report_writing(
                    question=query,
                    research_summary=research_summary,
                    report_type=self.report_type,
                    agent_role_prompt=self.role,
                    outline=outline,
                    agent_name=self.agent_name,
                    dir_path=self.dir_path,
                )
                break
            except Exception as e:
                await self._callback_manager.on_run_error(
                    tool_name=self.report_writing.description, error_information=str(e)
                )
                retry_count += 1
                if retry_count > MAX_RETRY:
                    raise Exception(f"Failed to conduct research for {query} after {MAX_RETRY} times.")
                continue
        await self._callback_manager.on_tool_end(self, tool=self.report_writing, response=report)
        await self._callback_manager.on_run_end(agent=self, agent_name=self.name, response=f"报告存储在{path}")
        return report, meta_data, paragraphs
