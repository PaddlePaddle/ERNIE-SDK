import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Type

from pydantic import Field

from erniebot_agent.agents.function_agent import FunctionAgent
from erniebot_agent.agents.schema import (
    NO_ACTION_STEP,
    AgentResponse,
    AgentStep,
    File,
    NoActionStep,
    PluginStep,
    ToolAction,
    ToolInfo,
    ToolResponse,
    ToolStep,
)
from erniebot_agent.memory.messages import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    Message,
    SearchInfo,
)
from erniebot_agent.prompt import PromptTemplate
from erniebot_agent.retrieval import BaizhongSearch
from erniebot_agent.tools.base import Tool
from erniebot_agent.tools.schema import ToolParameterView

INTENT_PROMPT = """检索结果:
{% for doc in documents %}
    第{{loop.index}}个段落: {{doc['content']}}
{% endfor %}
检索语句: {{query}}
请判断以上的检索结果和检索语句是否相关，并且有助于回答检索语句的问题。
请严格按照【JSON格式】输出。如果相关，则回复：{"is_relevant":true}，如果不相关，则回复：{"is_relevant":false}"""

RAG_PROMPT = """检索结果:
{% for doc in documents %}
    第{{loop.index}}个段落: {{doc['content']}}
{% endfor %}
检索语句: {{query}}
请根据以上检索结果回答检索语句的问题"""


_logger = logging.getLogger(__name__)


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


class FunctionAgentWithRetrieval(FunctionAgent):
    def __init__(
        self,
        knowledge_base: BaizhongSearch,
        top_k: int = 3,
        threshold: float = 0.0,
        token_limit: int = 3000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.knowledge_base = knowledge_base
        self.top_k = top_k
        self.threshold = threshold
        self.rag_prompt = PromptTemplate(RAG_PROMPT, input_variables=["documents", "query"])
        self.search_tool = KnowledgeBaseTool()
        self.token_limit = token_limit

    async def _run(self, prompt: str, files: Optional[Sequence[File]] = None) -> AgentResponse:
        results = await self._maybe_retrieval(prompt)
        if len(results["documents"]) > 0:
            # RAG branch
            tool_args = json.dumps({"query": prompt}, ensure_ascii=False)
            # on_tool_start callback
            await self._callback_manager.on_tool_start(
                agent=self, tool=self.search_tool, input_args=tool_args
            )
            try:
                docs = self._enforce_token_limit(results)
                step_input = HumanMessage(content=self.rag_prompt.format(query=prompt, documents=docs))
                chat_history: List[Message] = []
                chat_history.append(step_input)
                steps_taken: List[AgentStep] = []

                tool_ret_json = json.dumps(results, ensure_ascii=False)
                tool_resp = ToolResponse(json=tool_ret_json, input_files=[], output_files=[])
                steps_taken.append(
                    ToolStep(
                        info=ToolInfo(tool_name=self.search_tool.tool_name, tool_args=tool_args),
                        result=tool_resp.json,
                        input_files=tool_resp.input_files,
                        output_files=tool_resp.output_files,
                    )
                )
                llm_resp = await self._run_llm(
                    messages=chat_history,
                    functions=None,
                )
                output_message = llm_resp.message
                if output_message.search_info is None:
                    search_info = SearchInfo(results=[])
                    for index, item in enumerate(docs):
                        search_info["results"].append(
                            {
                                "index": index + 1,
                                "url": "",
                                "title": item["title"],
                            }
                        )
                    output_message.search_info = search_info
                else:
                    cur_index = len(output_message.search_info["results"])
                    for index, item in enumerate(docs):
                        output_message.search_info["results"].append(
                            {"index": cur_index + index + 1, "url": "", "title": item["title"]}
                        )
                chat_history.append(output_message)
            # Using on_tool_error here since retrieval is formatted as a tool
            except (Exception, KeyboardInterrupt) as e:
                await self._callback_manager.on_tool_error(agent=self, tool=self.search_tool, error=e)
                raise
            await self._callback_manager.on_tool_end(agent=self, tool=self.search_tool, response=tool_resp)
            response = self._create_finished_response(chat_history, steps_taken, curr_step=NO_ACTION_STEP)
            self.memory.add_message(chat_history[0])
            self.memory.add_message(chat_history[-1])
            return response
        else:
            _logger.info(
                f"Irrelevant retrieval results. Fallbacking to FunctionAgent for the query: {prompt}"
            )
            return await super()._run(prompt, files)

    def _enforce_token_limit(self, results):
        docs = []
        token_count = 0
        for doc in results["documents"]:
            num_tokens = len(doc["content"])
            if token_count + num_tokens > self.token_limit:
                _logger.warning(
                    "Retrieval results exceed token limit. Truncating retrieval results to "
                    f"under {self.token_limit} tokens"
                )
                break
            else:
                token_count += num_tokens
                docs.append(doc)
        return docs

    async def _maybe_retrieval(
        self,
        step_input,
    ):
        documents = self.knowledge_base.search(step_input, top_k=self.top_k, filters=None)
        documents = [item for item in documents if item["score"] > self.threshold]
        results = {}
        results["documents"] = documents
        return results


class FunctionAgentWithRetrievalTool(FunctionAgent):
    def __init__(self, knowledge_base: BaizhongSearch, top_k: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.knowledge_base = knowledge_base
        self.top_k = top_k
        self.intent_prompt = PromptTemplate(INTENT_PROMPT, input_variables=["documents", "query"])
        self.rag_prompt = PromptTemplate(RAG_PROMPT, input_variables=["documents", "query"])
        self.search_tool = KnowledgeBaseTool()

    async def _run(self, prompt: str, files: Optional[Sequence[File]] = None) -> AgentResponse:
        results = await self._maybe_retrieval(prompt)
        if results["is_relevant"] is True:
            # RAG
            chat_history: List[Message] = []
            steps_taken: List[AgentStep] = []

            tool_args = json.dumps({"query": prompt}, ensure_ascii=False)
            await self._callback_manager.on_tool_start(
                agent=self, tool=self.search_tool, input_args=tool_args
            )

            chat_history.append(HumanMessage(content=prompt))
            outputs = []
            for item in results["documents"]:
                outputs.append(
                    {
                        "id": item["id"],
                        "title": item["title"],
                        "document": item["content"],
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
            action = ToolAction(tool_name=self.search_tool.tool_name, tool_args=tool_args)
            # return response
            tool_ret_json = json.dumps({"documents": outputs}, ensure_ascii=False)
            next_step_input = FunctionMessage(name=action.tool_name, content=tool_ret_json)
            chat_history.append(next_step_input)
            tool_resp = ToolResponse(json=tool_ret_json, input_files=[], output_files=[])
            steps_taken.append(
                ToolStep(
                    info=ToolInfo(tool_name=self.search_tool.tool_name, tool_args=tool_args),
                    result=tool_resp.json,
                    input_files=tool_resp.input_files,
                    output_files=tool_resp.output_files,
                )
            )
            await self._callback_manager.on_tool_end(agent=self, tool=self.search_tool, response=tool_resp)

            num_steps_taken = 0
            while num_steps_taken < self.max_steps:
                curr_step, new_messages = await self._step(chat_history)
                chat_history.extend(new_messages)
                if not isinstance(curr_step, NoActionStep):
                    steps_taken.append(curr_step)
                if isinstance(curr_step, (NoActionStep, PluginStep)):  # plugin with action
                    response = self._create_finished_response(chat_history, steps_taken, curr_step)
                    self.memory.add_message(chat_history[0])
                    self.memory.add_message(chat_history[-1])
                    return response
                num_steps_taken += 1
            response = self._create_stopped_response(chat_history, steps_taken)
            return response
        else:
            _logger.info(
                f"Irrelevant retrieval results. Fallbacking to FunctionAgent for the query: {prompt}"
            )
            return await super()._run(prompt)

    async def _maybe_retrieval(
        self,
        step_input,
    ):
        documents = self.knowledge_base.search(step_input, top_k=self.top_k, filters=None)
        messages = [HumanMessage(content=self.intent_prompt.format(documents=documents, query=step_input))]
        response = await self._run_llm(messages)
        results = self._parse_results(response.message.content)
        results["documents"] = documents
        return results

    def _parse_results(self, results):
        left_index = results.find("{")
        right_index = results.rfind("}")
        if left_index == -1 or right_index == -1:
            # if invalid json, use FunctionAgent
            return {"is_relevant": False}
        try:
            return json.loads(results[left_index : right_index + 1])
        except Exception:
            # if invalid json, use FunctionAgent
            return {"is_relevant": False}


class FunctionAgentWithRetrievalScoreTool(FunctionAgent):
    def __init__(self, knowledge_base: BaizhongSearch, top_k: int = 3, threshold: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.knowledge_base = knowledge_base
        self.top_k = top_k
        self.threshold = threshold
        self.rag_prompt = PromptTemplate(RAG_PROMPT, input_variables=["documents", "query"])
        self.search_tool = KnowledgeBaseTool()

    async def _run(self, prompt: str, files: Optional[Sequence[File]] = None) -> AgentResponse:
        results = await self._maybe_retrieval(prompt)
        if len(results["documents"]) > 0:
            # RAG
            chat_history: List[Message] = []
            steps_taken: List[AgentStep] = []

            tool_args = json.dumps({"query": prompt}, ensure_ascii=False)
            await self._callback_manager.on_tool_start(
                agent=self, tool=self.search_tool, input_args=tool_args
            )
            chat_history.append(HumanMessage(content=prompt))
            outputs = []
            for item in results["documents"]:
                outputs.append(
                    {
                        "id": item["id"],
                        "title": item["title"],
                        "document": item["content"],
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
            action = ToolAction(tool_name=self.search_tool.tool_name, tool_args=tool_args)
            # return response
            tool_ret_json = json.dumps({"documents": outputs}, ensure_ascii=False)
            next_step_input = FunctionMessage(name=action.tool_name, content=tool_ret_json)
            chat_history.append(next_step_input)
            tool_resp = ToolResponse(json=tool_ret_json, input_files=[], output_files=[])
            steps_taken.append(
                ToolStep(
                    info=ToolInfo(tool_name=self.search_tool.tool_name, tool_args=tool_args),
                    result=tool_resp.json,
                    input_files=tool_resp.input_files,
                    output_files=tool_resp.output_files,
                )
            )
            await self._callback_manager.on_tool_end(agent=self, tool=self.search_tool, response=tool_resp)
            num_steps_taken = 0
            while num_steps_taken < self.max_steps:
                curr_step, new_messages = await self._step(chat_history)
                chat_history.extend(new_messages)
                if not isinstance(curr_step, NoActionStep):
                    steps_taken.append(curr_step)
                if isinstance(curr_step, (NoActionStep, PluginStep)):  # plugin with action
                    response = self._create_finished_response(chat_history, steps_taken, curr_step=curr_step)
                    self.memory.add_message(chat_history[0])
                    self.memory.add_message(chat_history[-1])
                    return response
                num_steps_taken += 1
            response = self._create_stopped_response(chat_history, steps_taken)
            # while num_steps_taken < self.max_steps:
            #     curr_step_output = await self._step(
            #         next_step_input, chat_history, actions_taken, files_involved
            #     )
            #     if curr_step_output is None:
            #         response = self._create_finished_response(chat_history, actions_taken, files_involved)
            #         self.memory.add_message(chat_history[0])
            #         self.memory.add_message(chat_history[-1])
            #         return response
            #     num_steps_taken += 1
            # # response = self._create_stopped_response(chat_history, actions_taken, files_involved)
            # self._create_stopped_response(chat_history, steps_taken)
            return response
        else:
            _logger.info(
                f"Irrelevant retrieval results. Fallbacking to FunctionAgent for the query: {prompt}"
            )
            return await super()._run(prompt)

    async def _maybe_retrieval(
        self,
        step_input,
    ):
        documents = self.knowledge_base.search(step_input, top_k=self.top_k, filters=None)
        documents = [item for item in documents if item["score"] > self.threshold]
        results = {}
        results["documents"] = documents
        return results
