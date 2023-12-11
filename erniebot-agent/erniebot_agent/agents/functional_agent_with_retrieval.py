import json
from typing import List, Optional

from erniebot_agent.agents import FunctionalAgent
from erniebot_agent.agents.schema import (
    AgentAction,
    AgentFile,
    AgentResponse,
    ToolResponse,
)
from erniebot_agent.file_io.base import File
from erniebot_agent.messages import AIMessage, FunctionMessage, HumanMessage, Message
from erniebot_agent.prompt import PromptTemplate
from erniebot_agent.retrieval import BaizhongSearch
from erniebot_agent.utils.logging import logger

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


class FunctionalAgentWithRetrieval(FunctionalAgent):
    def __init__(self, knowledge_base: BaizhongSearch, top_k: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.knowledge_base = knowledge_base
        self.top_k = top_k
        self.intent_prompt = PromptTemplate(INTENT_PROMPT, input_variables=["documents", "query"])
        self.rag_prompt = PromptTemplate(RAG_PROMPT, input_variables=["documents", "query"])

    async def _async_run(self, prompt: str, files: Optional[List[File]] = None) -> AgentResponse:
        results = await self._maybe_retrieval(prompt)
        if results["is_relevant"] is True:
            # RAG
            step_input = HumanMessage(
                content=self.rag_prompt.format(query=prompt, documents=results["documents"])
            )
            chat_history: List[Message] = []
            actions_taken: List[AgentAction] = []
            files_involved: List[AgentFile] = []

            chat_history.append(step_input)
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
        messages = [HumanMessage(content=self.intent_prompt.format(documents=documents, query=step_input))]
        response = await self._async_run_llm(messages)
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


class FunctionalAgentWithRetrievalTool(FunctionalAgent):
    def __init__(self, knowledge_base: BaizhongSearch, top_k: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.knowledge_base = knowledge_base
        self.top_k = top_k
        self.intent_prompt = PromptTemplate(INTENT_PROMPT, input_variables=["documents", "query"])
        self.rag_prompt = PromptTemplate(RAG_PROMPT, input_variables=["documents", "query"])

    async def _async_run(self, prompt: str, files: Optional[List[File]] = None) -> AgentResponse:
        results = await self._maybe_retrieval(prompt)
        if results["is_relevant"] is True:
            # RAG
            chat_history: List[Message] = []
            actions_taken: List[AgentAction] = []
            files_involved: List[AgentFile] = []

            tool_args = '{"query": "%s"}' % prompt
            tool = self._tool_manager.get_tool("BaizhongSearchTool")
            await self._callback_manager.on_tool_start(agent=self, tool=tool, input_args=tool_args)

            chat_history.append(HumanMessage(content=prompt))

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
                        "name": "BaizhongSearchTool",
                        "thoughts": "这是一个检索的需求，我需要在BaizhongSearchTool知识库中检索出与输入的query相关的段落，并返回给用户。",
                        "arguments": '{"query": "%s"}' % prompt,
                    },
                )
            )

            # Knowledge Retrieval Tool
            action = AgentAction(tool_name="BaizhongSearchTool", tool_args='{"query": "%s"}' % prompt)
            actions_taken.append(action)
            # return response
            tool_ret_json = json.dumps({"documents": outputs}, ensure_ascii=False)
            next_step_input = FunctionMessage(name=action.tool_name, content=tool_ret_json)
            tool_resp = ToolResponse(json=tool_ret_json, files=[])
            await self._callback_manager.on_tool_end(agent=self, tool=tool, response=tool_resp)

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
        self.intent_prompt = PromptTemplate(INTENT_PROMPT, input_variables=["documents", "query"])
        self.rag_prompt = PromptTemplate(RAG_PROMPT, input_variables=["documents", "query"])

    async def _async_run(self, prompt: str, files: Optional[List[File]] = None) -> AgentResponse:
        results = await self._maybe_retrieval(prompt)
        if len(results["documents"]) > 0:
            # RAG
            chat_history: List[Message] = []
            actions_taken: List[AgentAction] = []
            files_involved: List[AgentFile] = []

            tool_args = '{"query": "%s"}' % prompt
            tool = self._tool_manager.get_tool("BaizhongSearchTool")
            await self._callback_manager.on_tool_start(agent=self, tool=tool, input_args=tool_args)
            chat_history.append(HumanMessage(content=prompt))
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
                        "name": "BaizhongSearchTool",
                        "thoughts": "这是一个检索的需求，我需要在BaizhongSearchTool知识库中检索出与输入的query相关的段落，并返回给用户。",
                        "arguments": tool_args,
                    },
                )
            )

            # Knowledge Retrieval Tool
            action = AgentAction(tool_name="BaizhongSearchTool", tool_args=tool_args)
            actions_taken.append(action)
            # return response
            tool_ret_json = json.dumps({"documents": outputs}, ensure_ascii=False)
            next_step_input = FunctionMessage(name=action.tool_name, content=tool_ret_json)
            tool_resp = ToolResponse(json=tool_ret_json, files=[])
            await self._callback_manager.on_tool_end(agent=self, tool=tool, response=tool_resp)
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
