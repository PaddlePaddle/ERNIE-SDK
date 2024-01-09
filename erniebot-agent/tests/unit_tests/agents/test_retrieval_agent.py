from unittest import mock
from unittest.mock import MagicMock

import pytest

from erniebot_agent.agents import RetrievalAgent
from erniebot_agent.memory import AIMessage
from erniebot_agent.retrieval import BaizhongSearch
from erniebot_agent.tools.baizhong_tool import BaizhongSearchTool
from tests.unit_tests.agents.common_util import EXAMPLE_RESPONSE
from tests.unit_tests.testing_utils.mocks.mock_chat_models import (
    FakeERNIEBotWithPresetResponses,
)
from tests.unit_tests.testing_utils.mocks.mock_memory import FakeMemory


class FakeFewShotSearch:
    def search(self, query: str, top_k: int = 10, **kwargs):
        retrieval_results = [
            {
                "content": "电动汽车的品牌有哪些？各有什么特点？",
                "sub_queries": {
                    "sub_query_1": "当前市场上的主要电动汽车品牌。",
                    "sub_query_2": "每个品牌的电动汽车品牌的基本技术规格，如续航里程、充电速度等。",
                },
                "score": 0.5,
            }
        ]
        return retrieval_results


class FakeAbstractSearch:
    def search(self, query: str, top_k: int = 10, **kwargs):
        retrieval_results = [
            {
                "content": "住房和城乡建设部规章城市管理执法办法的摘要",
                "score": 0.5,
            }
        ]
        return retrieval_results


@pytest.mark.asyncio
async def test_retrieval_agent_run_few_shot():
    knowledge_base_name = "test"
    access_token = "your access token"
    knowledge_base_id = 111
    with mock.patch("requests.post") as my_mock:
        baizhong_db = BaizhongSearch(
            knowledge_base_name=knowledge_base_name,
            access_token=access_token,
            knowledge_base_id=knowledge_base_id if knowledge_base_id != "" else None,
        )
    search_tool = BaizhongSearchTool(
        name="city_design_management", description="提供城市设计管理办法的信息", db=baizhong_db, threshold=0.0
    )
    few_shot_retriever = FakeFewShotSearch()
    llm = FakeERNIEBotWithPresetResponses(
        responses=[
            AIMessage('{"sub_query_":"具体子问题1","sub_query_2":"具体子问题2"}', function_call=None),
            AIMessage("Text response", function_call=None),
        ]
    )
    agent = RetrievalAgent(
        knowledge_base=search_tool,
        llm=llm,
        few_shot_retriever=few_shot_retriever,
        tools=[],
        memory=FakeMemory(),
    )

    with mock.patch("requests.post") as my_mock:
        my_mock.return_value = MagicMock(status_code=200, json=lambda: EXAMPLE_RESPONSE)
        response = await agent.run("Hello, world!")

    assert response.text == "Text response"
    assert (
        response.chat_history[0].content
        == "检索结果:\n\n    第1个段落: 住房和城乡建设部规章城市管理执法办法\n\n    第2个段落: 城市管理执法主管部门应当定期开展执法人员的培训和考核。\n\n"
        "检索语句: Hello, world!\n请根据以上检索结果回答检索语句的问题"
    )
    assert response.chat_history[1].content == "Text response"

    assert response.steps[0].name == "few shot retriever"
    assert response.steps[1].name == "sub query results 0"
    assert response.steps[2].name == "sub query results 1"


@pytest.mark.asyncio
async def test_retrieval_agent_run_context_planning():
    knowledge_base_name = "test"
    access_token = "your access token"
    knowledge_base_id = 111
    with mock.patch("requests.post") as my_mock:
        baizhong_db = BaizhongSearch(
            knowledge_base_name=knowledge_base_name,
            access_token=access_token,
            knowledge_base_id=knowledge_base_id if knowledge_base_id != "" else None,
        )
    search_tool = BaizhongSearchTool(
        name="city_design_management", description="提供城市设计管理办法的信息", db=baizhong_db, threshold=0.0
    )

    llm = FakeERNIEBotWithPresetResponses(
        responses=[
            AIMessage('{"sub_query_":"具体子问题1","sub_query_2":"具体子问题2"}', function_call=None),
            AIMessage("Text response", function_call=None),
        ]
    )
    context_retriever = FakeAbstractSearch()
    agent = RetrievalAgent(
        knowledge_base=search_tool,
        llm=llm,
        context_retriever=context_retriever,
        tools=[],
        memory=FakeMemory(),
    )

    with mock.patch("requests.post") as my_mock:
        my_mock.return_value = MagicMock(status_code=200, json=lambda: EXAMPLE_RESPONSE)
        response = await agent.run("Hello, world!")

    assert response.text == "Text response"
    assert (
        response.chat_history[0].content
        == "检索结果:\n\n    第1个段落: 住房和城乡建设部规章城市管理执法办法\n\n    第2个段落: 城市管理执法主管部门应当定期开展执法人员的培训和考核。\n\n"
        "检索语句: Hello, world!\n请根据以上检索结果回答检索语句的问题"
    )
    assert response.chat_history[1].content == "Text response"

    assert response.steps[0].name == "context retriever"
    assert response.steps[1].name == "sub query results 0"
    assert response.steps[2].name == "sub query results 1"


@pytest.mark.asyncio
async def test_retrieval_agent_run_compressor():
    knowledge_base_name = "test"
    access_token = "your access token"
    knowledge_base_id = 111
    with mock.patch("requests.post") as my_mock:
        baizhong_db = BaizhongSearch(
            knowledge_base_name=knowledge_base_name,
            access_token=access_token,
            knowledge_base_id=knowledge_base_id if knowledge_base_id != "" else None,
        )
    search_tool = BaizhongSearchTool(
        name="city_design_management", description="提供城市设计管理办法的信息", db=baizhong_db, threshold=0.0
    )

    llm = FakeERNIEBotWithPresetResponses(
        responses=[
            AIMessage('{"sub_query_":"具体子问题1","sub_query_2":"具体子问题2"}', function_call=None),
            AIMessage("Sub query compress 1", function_call=None),
            AIMessage("Sub query compress 2", function_call=None),
            AIMessage("Text response", function_call=None),
        ]
    )
    agent = RetrievalAgent(
        knowledge_base=search_tool,
        llm=llm,
        tools=[],
        use_compressor=True,
        memory=FakeMemory(),
    )

    with mock.patch("requests.post") as my_mock:
        my_mock.return_value = MagicMock(status_code=200, json=lambda: EXAMPLE_RESPONSE)
        response = await agent.run("Hello, world!")

    assert response.text == "Text response"
    assert (
        response.chat_history[0].content == "检索结果:\n\n    第1个段落: Sub query compress 1\n\n    "
        "第2个段落: Sub query compress 2\n\n检索语句: Hello, world!\n请根据以上检索结果回答检索语句的问题"
    )
    assert response.chat_history[1].content == "Text response"
    assert response.steps[0].name == "sub query compressor 0"
    assert response.steps[1].name == "sub query compressor 1"
