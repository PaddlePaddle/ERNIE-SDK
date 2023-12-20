import json
from typing import Any, Dict, List, Optional, Type

from erniebot_agent.agents import FunctionalAgent
from erniebot_agent.agents.schema import (
    AgentAction,
    AgentFile,
    AgentResponse,
    ToolResponse,
)
from erniebot_agent.file_io.base import File
from erniebot_agent.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    Message,
    SystemMessage,
)
from erniebot_agent.prompt import PromptTemplate
from erniebot_agent.retrieval import BaizhongSearch
from erniebot_agent.tools.base import Tool
from erniebot_agent.tools.schema import ToolParameterView
from erniebot_agent.utils.logging import logger
from pydantic import Field

INTENT_PROMPT = """检索结果:
{% for doc in documents %}
    第{{loop.index}}个段落: {{doc['content_se']}}
{% endfor %}
检索语句: {{query}}
请判断以上的检索结果和检索语句是否相关，并且有助于回答检索语句的问题。
请严格按照【JSON格式】输出。如果相关，则回复：{"is_relevant":true}，如果不相关，则回复：{"is_relevant":false}"""

RAG_PROMPT = """检索结果:
{% for doc in documents %}
    第{{loop.index}}个段落: {{doc['content_se']}}
{% endfor %}
检索语句: {{query}}
请根据以上检索结果回答检索语句的问题"""


class KnowledgeBaseToolInputView(ToolParameterView):
    query: str = Field(description="查询语句")
    top_k: int = Field(description="返回结果数量")


class SearchResponseDocument(ToolParameterView):
    id: str = Field(description="检索结果的文本的id")
    title: str = Field(description="检索结果的标题")
    document: str = Field(description="检索结果的内容")


class KnowledgeBaseToolOutputView(ToolParameterView):
    documents: List[SearchResponseDocument] = Field(description="检索结果，内容和用户输入query相关的段落")


class KnowledgeBaseTool(Tool):
    tool_name: str = "KnowledgeBaseTool"
    description: str = "在知识库中检索与用户输入query相关的段落"
    input_type: Type[ToolParameterView] = KnowledgeBaseToolInputView
    ouptut_type: Type[ToolParameterView] = KnowledgeBaseToolOutputView

    def __init__(
        self,
    ) -> None:
        super().__init__()

    async def __call__(self, query: str, top_k: int = 3, filters: Optional[Dict[str, Any]] = None):
        return {"documents": "This is the fake search tool."}


class FunctionalAgentWithRetrieval(FunctionalAgent):
    def __init__(self, knowledge_base: BaizhongSearch, top_k: int = 3, threshold: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.knowledge_base = knowledge_base
        self.top_k = top_k
        self.threshold = threshold
        self.rag_prompt = PromptTemplate(RAG_PROMPT, input_variables=["documents", "query"])
        self.search_tool = KnowledgeBaseTool()

    async def _async_run(self, prompt: str, files: Optional[List[File]] = None) -> AgentResponse:
        results = await self._maybe_retrieval(prompt)
        if len(results["documents"]) > 0:
            # RAG branch
            tool_args = json.dumps({"query": prompt}, ensure_ascii=False)
            # on_tool_start callback
            await self._callback_manager.on_tool_start(
                agent=self, tool=self.search_tool, input_args=tool_args
            )
            step_input = HumanMessage(
                content=self.rag_prompt.format(query=prompt, documents=results["documents"])
            )
            chat_history: List[Message] = [step_input]
            actions_taken: List[AgentAction] = []
            files_involved: List[AgentFile] = []
            action = AgentAction(tool_name="KnowledgeBaseTool", tool_args=tool_args)
            actions_taken.append(action)

            # on_tool_end callback
            tool_ret_json = json.dumps(results, ensure_ascii=False)
            tool_resp = ToolResponse(json=tool_ret_json, files=[])
            await self._callback_manager.on_tool_end(agent=self, tool=self.search_tool, response=tool_resp)
            llm_resp = await self._async_run_llm_without_hooks(
                messages=chat_history,
                functions=None,
                system=self.system_message.content if self.system_message is not None else None,
            )
            output_message = llm_resp.message
            chat_history.append(output_message)
            response = self._create_finished_response(chat_history, actions_taken, files_involved)
            self.memory.add_message(chat_history[0])
            self.memory.add_message(chat_history[-1])
            return response
        else:
            logger.info(
                f"Irrelevant retrieval results. Fallbacking to FunctionalAgent for the query: {prompt}"
            )
            return await super()._async_run(prompt)

    async def _maybe_retrieval(
        self,
        step_input,
    ):
        documents = self.knowledge_base.search(step_input, top_k=self.top_k, filters=None)
        documents = [item for item in documents if item["score"] > self.threshold]
        results = {}
        results["documents"] = documents
        return results


class FunctionalAgentWithRetrievalTool(FunctionalAgent):
    def __init__(self, knowledge_base: BaizhongSearch, top_k: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.knowledge_base = knowledge_base
        self.top_k = top_k
        self.intent_prompt = PromptTemplate(INTENT_PROMPT, input_variables=["documents", "query"])
        self.rag_prompt = PromptTemplate(RAG_PROMPT, input_variables=["documents", "query"])
        self.search_tool = KnowledgeBaseTool()

    async def _async_run(self, prompt: str, files: Optional[List[File]] = None) -> AgentResponse:
        results = await self._maybe_retrieval(prompt)
        if results["is_relevant"] is True:
            # RAG
            chat_history: List[Message] = []
            actions_taken: List[AgentAction] = []
            files_involved: List[AgentFile] = []

            tool_args = json.dumps({"query": prompt}, ensure_ascii=False)
            await self._callback_manager.on_tool_start(
                agent=self, tool=self.search_tool, input_args=tool_args
            )

            chat_history.append(HumanMessage(content=prompt))

            step_input = HumanMessage(
                content=self.rag_prompt.format(query=prompt, documents=results["documents"])
            )
            fake_chat_history: List[Message] = [step_input]
            llm_resp = await self._async_run_llm_without_hooks(
                messages=fake_chat_history,
                functions=None,
                system=self.system_message.content if self.system_message is not None else None,
            )

            # Get RAG results
            output_message = llm_resp.message

            outputs = []
            for item in results["documents"]:
                outputs.append(
                    {
                        "id": item["id"],
                        "title": item["title"],
                        "document": item["content_se"],
                    }
                )

            chat_history.append(
                AIMessage(
                    content="",
                    function_call={
                        "name": "KnowledgeBaseTool",
                        "thoughts": "这是一个检索的需求，我需要在KnowledgeBaseTool知识库中检索出与输入的query相关的段落，并返回给用户。",
                        "arguments": tool_args,
                    },
                )
            )

            # Knowledge Retrieval Tool
            action = AgentAction(tool_name="KnowledgeBaseTool", tool_args=tool_args)
            actions_taken.append(action)
            # return response
            tool_ret_json = json.dumps({"documents": outputs}, ensure_ascii=False)
            # next_step_input = FunctionMessage(name=action.tool_name, content=tool_ret_json)
            next_step_input = FunctionMessage(name=action.tool_name, content=output_message.content)
            tool_resp = ToolResponse(json=tool_ret_json, files=[])
            await self._callback_manager.on_tool_end(agent=self, tool=self.search_tool, response=tool_resp)

            num_steps_taken = 0
            while num_steps_taken < self.max_steps:
                curr_step_output = await self._async_step(
                    next_step_input, chat_history, actions_taken, files_involved
                )
                if curr_step_output is None:
                    response = self._create_finished_response(chat_history, actions_taken, files_involved)
                    self.memory.add_message(chat_history[0])
                    self.memory.add_message(chat_history[-1])
                    return response
                num_steps_taken += 1
            response = self._create_stopped_response(chat_history, actions_taken, files_involved)
            return response
        else:
            logger.info(
                f"Irrelevant retrieval results. Fallbacking to FunctionalAgent for the query: {prompt}"
            )
            return await super()._async_run(prompt)

    async def _maybe_retrieval(
        self,
        step_input,
    ):
        documents = self.knowledge_base.search(step_input, top_k=self.top_k, filters=None)
        messages = [HumanMessage(content=self.intent_prompt.format(documents=documents, query=step_input))]
        response = await self._async_run_llm_without_hooks(messages)
        results = self._parse_results(response.message.content)
        results["documents"] = documents
        return results

    def _parse_results(self, results):
        left_index = results.find("{")
        right_index = results.rfind("}")
        if left_index == -1 or right_index == -1:
            # if invalid json, use Functional Agent
            return {"is_relevant": False}
        try:
            return json.loads(results[left_index : right_index + 1])
        except Exception:
            # if invalid json, use Functional Agent
            return {"is_relevant": False}


class FunctionalAgentWithRetrievalScoreTool(FunctionalAgent):
    def __init__(self, knowledge_base: BaizhongSearch, top_k: int = 3, threshold: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.knowledge_base = knowledge_base
        self.top_k = top_k
        self.threshold = threshold
        self.rag_prompt = PromptTemplate(RAG_PROMPT, input_variables=["documents", "query"])
        self.search_tool = KnowledgeBaseTool()

    async def _async_run(self, prompt: str, files: Optional[List[File]] = None) -> AgentResponse:
        results = await self._maybe_retrieval(prompt)
        if len(results["documents"]) > 0:
            # RAG
            chat_history: List[Message] = []
            actions_taken: List[AgentAction] = []
            files_involved: List[AgentFile] = []

            tool_args = json.dumps({"query": prompt}, ensure_ascii=False)
            await self._callback_manager.on_tool_start(
                agent=self, tool=self.search_tool, input_args=tool_args
            )

            outputs = []
            for item in results["documents"]:
                outputs.append(
                    {
                        "id": item["id"],
                        "title": item["title"],
                        "document": item["content_se"],
                    }
                )

            # return response
            tool_ret_json = json.dumps({"documents": outputs}, ensure_ascii=False)
            # Direct Prompt
            next_step_input = HumanMessage(content=f"问题：{prompt}，要求：请在第一步执行检索的操作,并且检索只允许调用一次")
            tool_resp = ToolResponse(json=tool_ret_json, files=[])
            await self._callback_manager.on_tool_end(agent=self, tool=self.search_tool, response=tool_resp)

            num_steps_taken = 0
            while num_steps_taken < self.max_steps:
                curr_step_output = await self._async_step(
                    next_step_input, chat_history, actions_taken, files_involved
                )
                if curr_step_output is None:
                    response = self._create_finished_response(chat_history, actions_taken, files_involved)
                    self.memory.add_message(chat_history[0])
                    self.memory.add_message(chat_history[-1])
                    return response
                num_steps_taken += 1
            response = self._create_stopped_response(chat_history, actions_taken, files_involved)
            return response
        else:
            logger.info(
                f"Irrelevant retrieval results. Fallbacking to FunctionalAgent for the query: {prompt}"
            )
            return await super()._async_run(prompt)

    async def _maybe_retrieval(
        self,
        step_input,
    ):
        documents = self.knowledge_base.search(step_input, top_k=self.top_k, filters=None)
        documents = [item for item in documents if item["score"] > self.threshold]
        results = {}
        results["documents"] = documents
        return results


class ContextAugmentedFunctionalAgent(FunctionalAgent):
    def __init__(self, knowledge_base: BaizhongSearch, top_k: int = 3, threshold: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.knowledge_base = knowledge_base
        self.top_k = top_k
        self.threshold = threshold
        self.rag_prompt = PromptTemplate(RAG_PROMPT, input_variables=["documents", "query"])
        self.search_tool = KnowledgeBaseTool()

    async def _async_run(self, prompt: str, files: Optional[List[File]] = None) -> AgentResponse:
        results = await self._maybe_retrieval(prompt)
        if len(results["documents"]) > 0:
            # RAG
            chat_history: List[Message] = []
            actions_taken: List[AgentAction] = []
            files_involved: List[AgentFile] = []

            tool_args = json.dumps({"query": prompt}, ensure_ascii=False)
            await self._callback_manager.on_tool_start(
                agent=self, tool=self.search_tool, input_args=tool_args
            )
            step_input = HumanMessage(
                content=self.rag_prompt.format(query=prompt, documents=results["documents"])
            )
            fake_chat_history: List[Message] = []
            fake_chat_history.append(step_input)
            llm_resp = await self._async_run_llm_without_hooks(
                messages=fake_chat_history,
                functions=None,
                system=self.system_message.content if self.system_message is not None else None,
            )

            # Get RAG results
            output_message = llm_resp.message

            outputs = []
            for item in results["documents"]:
                outputs.append(
                    {
                        "id": item["id"],
                        "title": item["title"],
                        "document": item["content_se"],
                    }
                )

            # return response
            tool_ret_json = json.dumps({"documents": outputs}, ensure_ascii=False)
            # 会有无限循环调用工具的问题
            next_step_input = HumanMessage(
                content=f"背景信息为：{output_message.content} \n 要求：选择相应的工具回答或者根据背景信息直接回答：{prompt}"
            )
            tool_resp = ToolResponse(json=tool_ret_json, files=[])
            await self._callback_manager.on_tool_end(agent=self, tool=self.search_tool, response=tool_resp)

            num_steps_taken = 0
            while num_steps_taken < self.max_steps:
                curr_step_output = await self._async_step(
                    next_step_input, chat_history, actions_taken, files_involved
                )
                if curr_step_output is None:
                    response = self._create_finished_response(chat_history, actions_taken, files_involved)
                    self.memory.add_message(chat_history[0])
                    self.memory.add_message(chat_history[-1])
                    return response
                num_steps_taken += 1
            response = self._create_stopped_response(chat_history, actions_taken, files_involved)
            return response
        else:
            logger.info(
                f"Irrelevant retrieval results. Fallbacking to FunctionalAgent for the query: {prompt}"
            )
            return await super()._async_run(prompt)

    async def _maybe_retrieval(
        self,
        step_input,
    ):
        documents = self.knowledge_base.search(step_input, top_k=self.top_k, filters=None)
        documents = [item for item in documents if item["score"] > self.threshold]
        results = {}
        results["documents"] = documents
        return results


QUERY_DECOMPOSITION = """请把下面的问题分解成子问题，每个子问题必须足够简单，要求：
1.严格按照【JSON格式】的形式输出：{'子问题1':'具体子问题1','子问题2':'具体子问题2'}
问题：{{prompt}} 子问题："""


OPENAI_RAG_PROMPT = """检索结果:
{% for doc in documents %}
    第{{loop.index}}个段落: {{doc['document']}}
{% endfor %}
检索语句: {{query}}
请根据以上检索结果回答检索语句的问题"""


class FunctionalAgentWithQueryPlanning(FunctionalAgent):
    def __init__(self, knowledge_base, top_k: int = 2, threshold: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.threshold = threshold
        self.system_message = SystemMessage(content="您是一个智能体，旨在回答有关知识库的查询。请始终使用提供的工具回答问题。不要依赖先验知识。")
        self.query_transform = PromptTemplate(QUERY_DECOMPOSITION, input_variables=["prompt"])
        self.knowledge_base = knowledge_base
        self.rag_prompt = PromptTemplate(OPENAI_RAG_PROMPT, input_variables=["documents", "query"])

    async def _async_run(self, prompt: str, files: Optional[List[File]] = None) -> AgentResponse:
        # chat_history: List[Message] = []
        actions_taken: List[AgentAction] = []
        files_involved: List[AgentFile] = []
        chat_history: List[Message] = []

        # 会有无限循环调用工具的问题
        # next_step_input = HumanMessage(
        #     content=f"请选择合适的工具来回答：{prompt}，如果需要的话，可以对把问题分解成子问题，然后每个子问题选择合适的工具回答。"
        # )
        next_step_input = HumanMessage(content=prompt)
        num_steps_taken = 0
        while num_steps_taken < self.max_steps:
            curr_step_output = await self._async_step(
                next_step_input, chat_history, actions_taken, files_involved
            )
            if curr_step_output is None:
                response = self._create_finished_response(chat_history, actions_taken, files_involved)
                self.memory.add_message(chat_history[0])
                self.memory.add_message(chat_history[-1])
                return response
            num_steps_taken += 1
        # TODO(wugaosheng): Add manual planning and execute
        # response = self._create_stopped_response(chat_history, actions_taken, files_involved)
        return await self.plan_and_execute(prompt, actions_taken, files_involved)

    async def plan_and_execute(self, prompt, actions_taken, files_involved):
        step_input = HumanMessage(content=self.query_transform.format(prompt=prompt))
        fake_chat_history: List[Message] = [step_input]
        llm_resp = await self._async_run_llm_without_hooks(
            messages=fake_chat_history,
            functions=None,
            system=self.system_message.content if self.system_message is not None else None,
        )
        output_message = llm_resp.message

        json_results = self._parse_results(output_message.content)
        sub_queries = json_results.values()
        retrieval_results = []
        duplicates = set()
        for query in sub_queries:
            documents = await self.knowledge_base(query, top_k=self.top_k, filters=None)
            docs = [item for item in documents["documents"]]
            for doc in docs:
                if doc["document"] not in duplicates:
                    duplicates.add(doc["document"])
                    retrieval_results.append(doc)
        step_input = HumanMessage(
            content=self.rag_prompt.format(query=prompt, documents=retrieval_results[:3])
        )
        chat_history: List[Message] = [step_input]
        llm_resp = await self._async_run_llm_without_hooks(
            messages=chat_history,
            functions=None,
            system=self.system_message.content if self.system_message is not None else None,
        )

        output_message = llm_resp.message
        chat_history.append(output_message)
        response = self._create_finished_response(chat_history, actions_taken, files_involved)
        self.memory.add_message(chat_history[0])
        self.memory.add_message(chat_history[-1])
        return response

    def _parse_results(self, results):
        left_index = results.find("{")
        right_index = results.rfind("}")
        if left_index == -1 or right_index == -1:
            # if invalid json, use Functional Agent
            return {"is_relevant": False}
        try:
            return json.loads(results[left_index : right_index + 1])
        except Exception:
            # if invalid json, use Functional Agent
            return {"is_relevant": False}
