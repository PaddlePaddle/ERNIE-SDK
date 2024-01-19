import json
from unittest import mock
from unittest.mock import MagicMock

import pytest

from erniebot_agent.agents import ContextAugmentedFunctionAgent
from erniebot_agent.memory import AIMessage, HumanMessage
from erniebot_agent.retrieval import BaizhongSearch
from erniebot_agent.tools.baizhong_tool import BaizhongSearchTool
from tests.unit_tests.agents.common_util import EXAMPLE_RESPONSE
from tests.unit_tests.testing_utils.components import CountingCallbackHandler
from tests.unit_tests.testing_utils.mocks.mock_chat_models import (
    FakeERNIEBotWithPresetResponses,
)
from tests.unit_tests.testing_utils.mocks.mock_memory import FakeMemory
from tests.unit_tests.testing_utils.mocks.mock_tool import FakeTool


@pytest.fixture(scope="module")
def identity_tool():
    return FakeTool(
        name="identity_tool",
        description="This tool simply forwards the input.",
        parameters={
            "type": "object",
            "properties": {
                "param": {
                    "type": "string",
                    "description": "Input parameter.",
                }
            },
        },
        responses={
            "type": "object",
            "properties": {
                "param": {
                    "type": "string",
                    "description": "Same as the input parameter.",
                }
            },
        },
        function=lambda param: {"param": param},
    )


@pytest.fixture(scope="module")
def no_input_no_output_tool():
    return FakeTool(
        name="no_input_no_output_tool",
        description="This tool takes no input parameters and returns no output parameters.",
        parameters={"type": "object", "properties": {}},
        responses={"type": "object", "properties": {}},
        function=lambda: {},
    )


@pytest.mark.asyncio
async def test_functional_agent_with_retrieval_retrieval_score_tool_callbacks(identity_tool):
    callback_handler = CountingCallbackHandler()
    knowledge_base_name = "test"
    access_token = "your access token"
    knowledge_base_id = 111
    search_db = BaizhongSearch(
        knowledge_base_name=knowledge_base_name,
        access_token=access_token,
        knowledge_base_id=knowledge_base_id if knowledge_base_id != "" else None,
    )
    search_tool = BaizhongSearchTool(db=search_db, description="城市建筑设计相关规定")
    llm = FakeERNIEBotWithPresetResponses(
        responses=[
            AIMessage("Text response", function_call=None),
            AIMessage('{"sub_query_1":"具体子问题1","sub_query_2":"具体子问题2"}', function_call=None),
            AIMessage("Text response", function_call=None),
            AIMessage("Text response", function_call=None),
        ]
    )
    agent = ContextAugmentedFunctionAgent(
        knowledge_base=search_tool,
        llm=llm,
        threshold=0.0,
        tools=[identity_tool],
        memory=FakeMemory(),
        callbacks=[callback_handler],
    )

    await agent.run_llm([HumanMessage("Hello, world!")])
    assert callback_handler.llm_starts == 1
    assert callback_handler.llm_ends == 1
    assert callback_handler.llm_errors == 0

    await agent.run_tool(identity_tool.tool_name, json.dumps({"param": "test"}))
    assert callback_handler.tool_starts == 1
    assert callback_handler.tool_ends == 1
    assert callback_handler.tool_errors == 0
    with mock.patch("requests.post") as my_mock:
        my_mock.return_value = MagicMock(status_code=200, json=lambda: EXAMPLE_RESPONSE)
        await agent.run("Hello, world!")
    assert callback_handler.run_starts == 1
    assert callback_handler.run_ends == 1
    assert callback_handler.run_errors == 0
    # call retrieval tool
    assert callback_handler.tool_starts == 2
    assert callback_handler.tool_ends == 2
    assert callback_handler.tool_errors == 0


@pytest.mark.asyncio
async def test_functional_agent_with_retrieval_retrieval_score_tool_run_retrieval(identity_tool):
    knowledge_base_name = "test"
    access_token = "your access token"
    knowledge_base_id = 111

    search_db = BaizhongSearch(
        knowledge_base_name=knowledge_base_name,
        access_token=access_token,
        knowledge_base_id=knowledge_base_id if knowledge_base_id != "" else None,
    )
    search_tool = BaizhongSearchTool(db=search_db, description="城市建筑设计相关规定")
    llm = FakeERNIEBotWithPresetResponses(
        responses=[
            AIMessage('{"sub_query_1":"具体子问题1","sub_query_2":"具体子问题2"}', function_call=None),
            AIMessage("Text response", function_call=None),
            AIMessage("Text response", function_call=None),
        ]
    )
    agent = ContextAugmentedFunctionAgent(
        knowledge_base=search_tool,
        llm=llm,
        threshold=0.0,
        tools=[identity_tool],
        memory=FakeMemory(),
    )
    # Test retrieval success
    with mock.patch("requests.post") as my_mock:
        my_mock.return_value = MagicMock(status_code=200, json=lambda: EXAMPLE_RESPONSE)
        response = await agent.run("Hello, world!")
    assert response.text == "Text response"
    # HumanMessage
    assert (
        response.chat_history[0].content
        == "背景信息为：Text response \n 给定背景信息，而不是先验知识，选择相应的工具回答或者根据背景信息直接回答问题：Hello, world!"
    )
    # AIMessage
    assert response.chat_history[1].content == "Text response"

    assert len(response.steps) == 1
    assert response.steps[0].info == {
        "tool_name": "KnowledgeBaseTool",
        "tool_args": '{"query": "Hello, world!"}',
    }
    assert (
        response.steps[0].result
        == '[{"id": "495735246643269", "content": "住房和城乡建设部规章城市管理执法办法", "title": "城市管理执法办法.pdf", '
        '"score": 0.01162862777709961}, {"id": "495735246643270", "content": "城市管理执法主管部门应当定期开展执法人员的培训和考核。",'
        ' "title": "城市管理执法办法.pdf", "score": 0.011362016201019287}]'
    )

    # Test retrieval failed
    llm = FakeERNIEBotWithPresetResponses(
        responses=[
            AIMessage('{"sub_query_1":"具体子问题1","sub_query_2":"具体子问题2"}', function_call=None),
            AIMessage("Text response", function_call=None),
            AIMessage("Text response", function_call=None),
        ]
    )
    agent = ContextAugmentedFunctionAgent(
        knowledge_base=search_tool,
        llm=llm,
        threshold=0.1,
        tools=[identity_tool],
        memory=FakeMemory(),
    )
    with mock.patch("requests.post") as my_mock:
        my_mock.return_value = MagicMock(status_code=200, json=lambda: EXAMPLE_RESPONSE)
        response = await agent.run("Hello, world!")
    assert response.text == "Text response"
    # HumanMessage
    assert response.chat_history[0].content == "Hello, world!"
    # AIMessage
    assert response.chat_history[1].content == "Text response"
