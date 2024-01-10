import json
from typing import Dict, List, Optional, Sequence

from erniebot_agent.agents.agent import Agent
from erniebot_agent.agents.schema import AgentResponse, AgentStep
from erniebot_agent.file import File
from erniebot_agent.memory.messages import HumanMessage, Message
from erniebot_agent.prompt import PromptTemplate

GRAPH_PROMPT = """id: 查询的唯一id
dependencies: 在我们可以提出问题之前需要回答的子问题列表。当可能有未知的事情时，我们使用子查询，并且我们需要提出多个问题来得到答案。依赖项必须仅是其他查询。
question: 我们正在使用问答系统提出的问题，如果我们提出多个问题，那么这个问题也是通过提供子问题的答案来提出的。
example:
##
query: 美国和姚明的祖国的人口有什么区别？
query_graph: {'query_graph': [{'dependencies': [],
                    'id': 1,
                    'question': "确定姚明的祖国"},
                    {'dependencies': [],
                    'id': 2,
                    'question': '查找美国的人口'},
                    {'dependencies': [1],
                    'id': 3,
                    'question': "查找姚明祖国的人口"},
                    {'dependencies': [2, 3],
                    'id': 4,
                    'question': "分析加拿大和姚明祖国之间的人口差异"}]}
##
query: {{query}}
query_graph:
"""


RAG_PROMPT = """
先验信息:
{% for doc in pre_context %}
    {{doc}}
{% endfor %}
检索结果:
{% for doc in documents %}
    第{{loop.index}}个段落: {{doc['content']}}
{% endfor %}
检索语句: {{query}}
请根据以上检索结果回答检索语句的问题"""


class DAGRetrievalAgent(Agent):
    def __init__(
        self,
        knowledge_base,
        top_k: int = 2,
        threshold: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.query_transform = PromptTemplate(GRAPH_PROMPT, input_variables=["query"])
        self.rag_prompt = PromptTemplate(RAG_PROMPT, input_variables=["documents", "query", "pre_context"])

    async def _run(self, prompt: str, files: Optional[Sequence[File]] = None) -> AgentResponse:
        steps_taken: List[AgentStep] = []

        steps_input = HumanMessage(content=self.query_transform.format(query=prompt))
        chat_history: List[Message] = [steps_input]
        # Query planning
        llm_resp = await self.run_llm(messages=[steps_input])
        output_message = llm_resp.message
        query_graph = self._parse_results(output_message.content)
        retrieval_results = await self.execute(query_graph)

        # Answer generation
        step_input = HumanMessage(content=self.rag_prompt.format(query=prompt, documents=retrieval_results))
        chat_history: List[Message] = [step_input]
        llm_resp = await self.run_llm(
            messages=chat_history,
        )

        chat_history.append(output_message)
        self.memory.add_message(chat_history[0])
        self.memory.add_message(chat_history[-1])
        response = self._create_finished_response(chat_history, steps_taken)
        return response

    async def execute(self, query_graph: Dict):
        contexts = {}
        for item in query_graph["query_graph"]:
            sub_query = item["question"]
            idx = item["id"]
            dependencies = item["dependencies"]
            # sub query search
            documents = await self.knowledge_base(sub_query, top_k=self.top_k, filters=None)

            # SubQuery Answer generation
            cur_contexts = []
            for dp in dependencies:
                cur_context = contexts[dp]
                cur_contexts.append(cur_context)
            step_input = HumanMessage(
                content=self.rag_prompt.format(query=sub_query, pre_context=cur_context, documents=documents)
            )
            chat_history: List[Message] = [step_input]
            llm_resp = await self.run_llm(
                messages=chat_history,
            )

            contexts[idx] = {"query": sub_query, "response": llm_resp.context}
        return contexts.values()

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

    def _parse_results(self, results):
        left_index = results.find("{")
        right_index = results.rfind("}")
        return json.loads(results[left_index : right_index + 1])
