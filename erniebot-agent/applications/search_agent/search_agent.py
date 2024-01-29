from erniebot_agent.chat_models.erniebot import BaseERNIEBot
from typing import Any, ClassVar, Dict, List, Optional
from erniebot_agent.prompt import PromptTemplate
from retrieval import CustomSearch
from erniebot_agent.memory import HumanMessage, Message, SystemMessage
import asyncio
from erniebot_agent.chat_models import ERNIEBot
from utils import JsonUtil
from collections import defaultdict

KEYWORDS = '''请提取文本中涉及的概念，年份，数量等实体，并将输出到json列表中.
json格式是:{"keywords": ["关键词1","关键词2"]}。
{% if documents|length > 0 %}
示例:
##
{% for doc in documents %}
 文本：{{doc['query']}}
 输出：{{doc['data']}}
{% endfor %}
##
{% endif %}
文本:{{query}}
输出：
'''

class SparseSearchAgent(JsonUtil):
    def __init__(self, llm: BaseERNIEBot, retrieval: CustomSearch, few_shots: List[Dict]=[], join_mode: str="reciprocal_rank_fusion"):
        self.llm = llm
        self.prompt = PromptTemplate(KEYWORDS, input_variables=["query", "documents"])
        self.retrieval = retrieval
        self.few_shots = few_shots
        self.join_mode = join_mode
        self.query_repeat = 5


    async def run(self, query: str) -> Dict:
        agent_resp = await self._run(query)
        return agent_resp

    async def _run(self, query, top_k_join=10):
        # TODO(wugaosheng): Add llm output to generate search keywords
        new_query = await self.query_expansion(query)
        # Add original query to avoid low-quality expanded keywords
        retrieval_results = []  
        expanded_queries = [query, new_query]
        for query in  expanded_queries:
            results = self.retrieval.search(query)
            retrieval_results.append(results["Data"]["results"])

        # Ranking by fusion score
        if self.join_mode == "reciprocal_rank_fusion":
            scores_map = self._calculate_rrf(retrieval_results)
        sorted_docs = sorted(scores_map.items(), key=lambda d: d[1], reverse=True)
        results = [inp for inp in retrieval_results]
        document_map = {doc['id']: doc for result in results for doc in result}
        docs = []
        for (title, score) in sorted_docs[:top_k_join]:
            doc = document_map[title]
            doc['score'] = score
            docs.append(doc)
        return docs

    async def query_expansion(self,query):
        content = self.prompt.format(query=query, documents=self.few_shots)
        messages: List[Message] = [HumanMessage(content)]
        response = await self.llm.chat(messages, response_format="json_object",enable_human_clarify=True)
        keyword_result  = self.parse_json(response.content)
        if "pageSize" in keyword_result:
            top_k_join = int(keyword_result["pageSize"])
        new_query = self.create_new_query(keyword_result["keywords"])
        return new_query

    def create_new_query(self, keywords):
        new_query = ','.join(keywords)
        # Repeat query to increase term weights
        new_query = new_query * self.query_repeat
        return new_query

    def _calculate_rrf(self, results):
        """
        Calculates the reciprocal rank fusion. The constant K is set to 61 (60 was suggested by the original paper,
        plus 1 as python lists are 0-based and the paper used 1-based ranking).
        """
        K = 61

        scores_map = defaultdict(int)
        for result in results:
            for rank, doc in enumerate(result):
                scores_map[doc['id']] += 1 / (K + rank)

        return scores_map
