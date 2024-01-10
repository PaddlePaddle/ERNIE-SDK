import json
from typing import List, Optional, Sequence

from erniebot_agent.agents.agent import Agent
from erniebot_agent.agents.schema import AgentResponse, AgentStep
from erniebot_agent.file import File
from erniebot_agent.memory.messages import HumanMessage, Message
from erniebot_agent.prompt import PromptTemplate

ZERO_SHOT_QUERY_DECOMPOSITION = """请把下面的问题分解成子问题，每个子问题必须足够简单，要求：
1.严格按照【JSON格式】的形式输出：{"sub_query_1":"具体子问题1","sub_query_2":"具体子问题2"}
问题：{{query}} 子问题："""

FEW_SHOT_QUERT_DECOMPOSITION = """请把下面的问题分解成子问题,
严格按照【JSON格式】的形式输出：{"sub_query_1":"具体子问题1","sub_query_2":"具体子问题2"}
示例:
##
{% for doc in documents %}
 问题：{{doc['content']}}
 子问题：{{doc['sub_queries']}}
{% endfor %}
##
问题：{{query}}
子问题:
"""

RAG_PROMPT = """检索结果:
{% for doc in documents %}
    第{{loop.index}}个段落: {{doc['content']}}
{% endfor %}
检索语句: {{query}}
请根据以上检索结果回答检索语句的问题"""


CONTENT_COMPRESSOR = """针对以下问题和背景，提取背景中与回答问题相关的任何部分，并原样保留。如果背景中没有与问题相关的部分，则返回{no_output_str}。
记住，不要编辑提取的背景部分。
> 问题: {{query}}
> 背景:
>>>
{{context}}
>>>
提取的相关部分:"""

CONTEXT_QUERY_DECOMPOSITION = """
{{context}} 请根据上述背景信息把下面的问题分解成子问题，每个子问题必须足够简单，要求：
严格按照【JSON格式】的形式输出：{"sub_query_1":"具体子问题1","sub_query_2":"具体子问题2"}
问题：{{query}} 子问题：
"""


class FewShotSearch:
    def __init__(self, db):
        self.db = db

    def search(self, query: str, top_k: int = 10, **kwargs):
        docs = self.db.similarity_search_with_relevance_scores(query, top_k)
        retrieval_results = []
        for doc, score in docs:
            retrieval_results.append(
                {"content": doc.page_content, "sub_queries": doc.metadata["sub_queries"], "score": score}
            )
        return retrieval_results


class ContextSearch:
    def __init__(self, db):
        self.db = db

    def search(self, query: str, top_k: int = 10, **kwargs):
        docs = self.db.similarity_search_with_relevance_scores(query, top_k)
        retrieval_results = []
        for doc, score in docs:
            retrieval_results.append({"content": doc.page_content, "score": score})
        return retrieval_results


class RetrievalAgent(Agent):
    def __init__(
        self,
        knowledge_base,
        few_shot_retriever: Optional[FewShotSearch] = None,
        context_retriever: Optional[ContextSearch] = None,
        top_k: int = 2,
        threshold: float = 0.1,
        use_compressor: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.threshold = threshold

        self.knowledge_base = knowledge_base
        self.few_shot_retriever = few_shot_retriever
        self.context_retriever = context_retriever
        if self.few_shot_retriever and self.context_retriever:
            raise Exception("Few shot retriever and context retriever shouldn't be used simutaneously")
        if few_shot_retriever:
            self.query_transform = PromptTemplate(
                FEW_SHOT_QUERT_DECOMPOSITION, input_variables=["query", "documents"]
            )
        elif self.context_retriever:
            self.query_transform = PromptTemplate(CONTEXT_QUERY_DECOMPOSITION, input_variables=["context", "query"])
        else:
            self.query_transform = PromptTemplate(ZERO_SHOT_QUERY_DECOMPOSITION, input_variables=["query"])
        self.rag_prompt = PromptTemplate(RAG_PROMPT, input_variables=["documents", "query"])
        self.use_compressor = use_compressor
        self.compressor = PromptTemplate(CONTENT_COMPRESSOR, input_variables=["context", "query"])
        

    async def _run(self, prompt: str, files: Optional[Sequence[File]] = None) -> AgentResponse:
        steps_taken: List[AgentStep] = []
        if self.few_shot_retriever:
            # Get few shot examples
            few_shots = self.few_shot_retriever.search(prompt, 3)
            steps_input = HumanMessage(
                content=self.query_transform.format(query=prompt, documents=few_shots)
            )
            steps_taken.append(
                AgentStep(info={"query": prompt, "name": "few shot retriever"}, result=few_shots)
            )
        elif self.context_retriever:
            res = self.context_retriever.search(prompt, 3)

            context = [item["content"] for item in res]
            steps_input = HumanMessage(
                content=self.query_transform.format(query=prompt, context="\n".join(context))
            )
            steps_taken.append(AgentStep(info={"query": prompt, "name": "context retriever"}, result=res))
        else:
            steps_input = HumanMessage(content=self.query_transform.format(query=prompt))
        # Query planning
        llm_resp = await self.run_llm(
            messages=[steps_input],
        )
        output_message = llm_resp.message
        json_results = self._parse_results(output_message.content)
        sub_queries = json_results.values()
        # Sub query execution
        retrieval_results = await self.execute(sub_queries, steps_taken)

        # Answer generation
        step_input = HumanMessage(content=self.rag_prompt.format(query=prompt, documents=retrieval_results))
        chat_history: List[Message] = [step_input]
        llm_resp = await self.run_llm(
            messages=chat_history,
        )

        output_message = llm_resp.message
        chat_history.append(output_message)
        self.memory.add_message(chat_history[0])
        self.memory.add_message(chat_history[-1])
        response = self._create_finished_response(chat_history, steps_taken)
        return response

    async def execute(self, sub_queries, steps_taken: List[AgentStep]):
        retrieval_results = []
        if self.use_compressor:
            for idx, query in enumerate(sub_queries):
                documents = await self.knowledge_base(query, top_k=self.top_k, filters=None)
                docs = [item for item in documents["documents"]]
                context = "\n".join([item["content"] for item in docs])
                llm_resp = await self.run_llm(
                    messages=[HumanMessage(content=self.compressor.format(query=query, context=context))]
                )
                # Parse Compressed results
                output_message = llm_resp.message
                compressed_data = docs[0]
                compressed_data["sub_query"] = query
                compressed_data["content"] = output_message.content
                retrieval_results.append(compressed_data)
                steps_taken.append(
                    AgentStep(
                        info={"query": query, "name": f"sub query compressor {idx}"}, result=compressed_data
                    )
                )
        else:
            duplicates = set()
            for idx, query in enumerate(sub_queries):
                documents = await self.knowledge_base(query, top_k=self.top_k, filters=None)
                docs = [item for item in documents["documents"]]
                steps_taken.append(
                    AgentStep(info={"query": query, "name": f"sub query results {idx}"}, result=documents)
                )
                for doc in docs:
                    if doc["content"] not in duplicates:
                        duplicates.add(doc["content"])
                        retrieval_results.append(doc)
            retrieval_results = retrieval_results[:3]
        return retrieval_results

    def _parse_results(self, results):
        left_index = results.find("{")
        right_index = results.rfind("}")
        return json.loads(results[left_index : right_index + 1])

    def _create_finished_response(
        self,
        chat_history: List[Message],
        steps: List[AgentStep],
    ) -> AgentResponse:
        last_message = chat_history[-1]
        return AgentResponse(
            text=last_message.content,
            chat_history=chat_history,
            steps=steps,
            status="FINISHED",
        )
