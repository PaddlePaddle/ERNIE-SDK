import pytest
from langchain.docstore.document import Document

from erniebot_agent.tools.langchain_retrieval_tool import LangChainRetrievalTool


class FakeSearch:
    def similarity_search_with_relevance_scores(self, query: str, top_k: int = 10, **kwargs):
        doc = (Document(page_content="电动汽车的品牌有哪些？各有什么特点？"), 0.5)
        retrieval_results = [doc]

        return retrieval_results


@pytest.fixture(scope="module")
def tool():
    db = FakeSearch()
    return LangChainRetrievalTool(db)


def test_schema(tool):
    function_call_schema = tool.function_call_schema()
    assert function_call_schema["description"] == LangChainRetrievalTool.description


@pytest.mark.asyncio
async def test_tool(tool):
    results = await tool(query="This is a test query")
    assert results == {"documents": [{"content": "电动汽车的品牌有哪些？各有什么特点？", "score": 0.5, "meta": {}}]}
