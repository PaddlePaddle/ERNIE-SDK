import json
from unittest import mock
from unittest.mock import MagicMock

import pytest
from tests.unit_tests.agents.common_util import EXAMPLE_RESPONSE
from tests.unit_tests.testing_utils.components import CountingCallbackHandler
from tests.unit_tests.testing_utils.mocks.mock_chat_models import (
    FakeChatModelWithPresetResponses,
    FakeSimpleChatModel,
)
from tests.unit_tests.testing_utils.mocks.mock_memory import FakeMemory
from tests.unit_tests.testing_utils.mocks.mock_tool import FakeTool

from erniebot_agent.agents import FunctionalAgentWithRetrievalTool
from erniebot_agent.memory import AIMessage, HumanMessage
from erniebot_agent.retrieval import BaizhongSearch


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
async def test_functional_agent_with_retrieval_tool_tool_callbacks(identity_tool):
    callback_handler = CountingCallbackHandler()
    knowledge_base_name = "test"
    access_token = "your access token"
    knowledge_base_id = 111
    search_db = BaizhongSearch(
        knowledge_base_name=knowledge_base_name,
        access_token=access_token,
        knowledge_base_id=knowledge_base_id if knowledge_base_id != "" else None,
    )
    agent = FunctionalAgentWithRetrievalTool(
        knowledge_base=search_db,
        llm=FakeSimpleChatModel(),
        tools=[identity_tool],
        memory=FakeMemory(),
        callbacks=[callback_handler],
    )

    await agent._async_run_llm([HumanMessage("Hello, world!")])
    assert callback_handler.llm_starts == 1
    assert callback_handler.llm_ends == 1
    assert callback_handler.llm_errors == 0

    await agent._async_run_tool(identity_tool.tool_name, json.dumps({"param": "test"}))
    assert callback_handler.tool_starts == 1
    assert callback_handler.tool_ends == 1
    assert callback_handler.tool_errors == 0
    with mock.patch("requests.post") as my_mock:
        my_mock.return_value = MagicMock(status_code=200, json=lambda: EXAMPLE_RESPONSE)
        await agent.async_run("Hello, world!")
    assert callback_handler.run_starts == 1
    assert callback_handler.run_ends == 1
    assert callback_handler.tool_starts == 2
    assert callback_handler.tool_ends == 2
    assert callback_handler.tool_errors == 0


@pytest.mark.asyncio
async def test_functional_agent_with_retrieval_tool_run_retrieval_tool(identity_tool):
    knowledge_base_name = "test"
    access_token = "your access token"
    knowledge_base_id = 111

    search_db = BaizhongSearch(
        knowledge_base_name=knowledge_base_name,
        access_token=access_token,
        knowledge_base_id=knowledge_base_id if knowledge_base_id != "" else None,
    )
    llm = FakeChatModelWithPresetResponses(
        responses=[
            AIMessage('{"is_relevant":true}', function_call=None),
            AIMessage("Text response", function_call=None),
        ]
    )
    agent = FunctionalAgentWithRetrievalTool(
        knowledge_base=search_db, llm=llm, tools=[identity_tool], memory=FakeMemory()
    )

    # Test retrieval success
    with mock.patch("requests.post") as my_mock:
        my_mock.return_value = MagicMock(status_code=200, json=lambda: EXAMPLE_RESPONSE)
        response = await agent.async_run("Hello, world!")

    assert response.text == "Text response"
    # HumanMessage
    assert response.chat_history[0].content == "Hello, world!"
    # AIMessage
    assert response.chat_history[1].function_call == {
        "name": "KnowledgeBaseTool",
        "thoughts": "这是一个检索的需求，我需要在KnowledgeBaseTool知识库中检索出与输入的query相关的段落，并返回给用户。",
        "arguments": '{"query": "Hello, world!"}',
    }
    # FunctionMessage
    assert response.chat_history[2].name == "KnowledgeBaseTool"
    assert (
        response.chat_history[2].content
        == '{"documents": [{"id": "495735246643269", "title": "城市管理执法办法.pdf", '
        '"document": "住房和城乡建设部规章城市管理执法办法"}, {"id": "495735246643270", '
        '"title": "城市管理执法办法.pdf", "document": "城市管理执法主管部门应当定期开展执法人员的培训和考核。"}]}'
    )
    # AIMessage
    assert response.chat_history[3].content == "Text response"

    llm = FakeChatModelWithPresetResponses(
        responses=[
            AIMessage('{"is_relevant":false}', function_call=None),
            AIMessage("Text response", function_call=None),
        ]
    )
    agent = FunctionalAgentWithRetrievalTool(
        knowledge_base=search_db, llm=llm, tools=[identity_tool], memory=FakeMemory()
    )
    # Test retrieval failed
    with mock.patch("requests.post") as my_mock:
        my_mock.return_value = MagicMock(status_code=200, json=lambda: EXAMPLE_RESPONSE)
        response = await agent.async_run("Hello, world!")
    assert response.text == "Text response"
    # HumanMessage
    assert response.chat_history[0].content == "Hello, world!"
    # AIMessage
    assert response.chat_history[1].content == "Text response"
