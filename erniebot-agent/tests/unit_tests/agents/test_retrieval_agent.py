from unittest import mock
from unittest.mock import MagicMock

import pytest

from erniebot_agent.agents import DAGRetrievalAgent, SelfAskRetrievalAgent
from erniebot_agent.memory import AIMessage
from erniebot_agent.retrieval import BaizhongSearch
from erniebot_agent.tools.baizhong_tool import BaizhongSearchTool
from tests.unit_tests.agents.common_util import EXAMPLE_RESPONSE
from tests.unit_tests.testing_utils.mocks.mock_chat_models import (
    FakeERNIEBotWithPresetResponses,
)
from tests.unit_tests.testing_utils.mocks.mock_memory import FakeMemory

FAKE_PROMPT = """
{"query_graph": [{"dependencies": [],
      "id": 1,
      "question": "确定姚明的祖国"},
    {"dependencies": [1],
      "id": 2,
      "question": "查找美国的人口"}]}
"""


@pytest.mark.asyncio
async def test_dag_retrieval_agent_run():
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
            AIMessage(FAKE_PROMPT, function_call=None),
            AIMessage("Text sub query response1", function_call=None),
            AIMessage("Text sub query response2", function_call=None),
            AIMessage("Text response", function_call=None),
        ]
    )
    agent = DAGRetrievalAgent(
        knowledge_base=search_tool,
        llm=llm,
        top_k=3,
        tools=[],
        memory=FakeMemory(),
    )
    with mock.patch("requests.post") as my_mock:
        my_mock.return_value = MagicMock(status_code=200, json=lambda: EXAMPLE_RESPONSE)
        response = await agent.run("Hello, world!")

    assert response.text == "Text response"
    assert (
        response.chat_history[0].content
        == "检索结果:\n\n    第1个子query: 确定姚明的祖国, 搜索结果：Text sub query response1\n\n    "
        "第2个子query: 查找美国的人口, 搜索结果：Text sub query response2\n\n检索语句: Hello, world!\n"
        "请根据以上子query的搜索结果提供的信息回答检索语句的问题."
    )
    assert response.chat_history[1].content == "Text response"
    assert response.steps[0].info == {
        "query": {"dependencies": [], "id": 1, "question": "确定姚明的祖国"},
        "name": "sub query results 0",
    }
    assert response.steps[1].info == {
        "query": {"dependencies": [1], "id": 2, "question": "查找美国的人口"},
        "name": "sub query results 1",
    }


FAKE_ASK_PROMPT = """
No answer provided
"""


@pytest.mark.asyncio
async def test_selfask_retrieval_agent_run():
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
            AIMessage(FAKE_ASK_PROMPT, function_call=None),
            AIMessage(
                '{"info":"不能直接回答该问题","accept": false,"sub query": ["子问题1","子问题2"]}', function_call=None
            ),
            AIMessage("子问题1的结果", function_call=None),
            AIMessage("子问题2的结果", function_call=None),
            AIMessage('{"info":"能正确回答该问题","accept": true}', function_call=None),
            AIMessage("Text response", function_call=None),
        ]
    )

    agent = SelfAskRetrievalAgent(
        knowledge_base=search_tool,
        llm=llm,
        top_k=3,
        tools=[],
        memory=FakeMemory(),
    )
    with mock.patch("requests.post") as my_mock:
        my_mock.return_value = MagicMock(status_code=200, json=lambda: EXAMPLE_RESPONSE)
        response = await agent.run("Hello, world!")

    assert response.text == "Text response"
    assert (
        response.chat_history[0].content
        == "检索结果:\n\n    \nNo answer provided\n\n\n    子问题1的结果\n\n    子问题2的结果\n\n"
        "检索语句: Hello, world!\n请根据以上子query的搜索结果提供的信息回答检索语句的问题."
    )
    assert response.chat_history[1].content == "Text response"
    assert response.steps[0].info == {"query": ["Hello, world!"], "name": "sub query results 0"}
    assert response.steps[1].info == {"query": ["子问题1", "子问题2"], "name": "sub query results 1"}
