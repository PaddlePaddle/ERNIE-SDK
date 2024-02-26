import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from erniebot_agent.tools.llama_index_retrieval_tool import LlamaIndexRetrievalTool


class FakeRetrieval:
    def retrieve(self, query: str):
        doc = NodeWithScore(node=TextNode(text="电动汽车的品牌有哪些？各有什么特点？"), score=0.5)
        retrieval_results = [doc]
        return retrieval_results


class FakeSearch:
    def as_retriever(self, similarity_top_k: int = 10, **kwargs):
        return FakeRetrieval()


@pytest.fixture(scope="module")
def tool():
    db = FakeSearch()
    return LlamaIndexRetrievalTool(db)


def test_schema(tool):
    function_call_schema = tool.function_call_schema()
    assert function_call_schema["description"] == LlamaIndexRetrievalTool.description


@pytest.mark.asyncio
async def test_tool(tool):
    results = await tool(query="This is a test query")
    assert results == {"documents": [{"content": "电动汽车的品牌有哪些？各有什么特点？", "score": 0.5, "meta": {}}]}
