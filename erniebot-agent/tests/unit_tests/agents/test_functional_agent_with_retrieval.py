import json
from unittest import mock
from unittest.mock import MagicMock

import pytest
from tests.unit_tests.agents.common_util import EXAMPLE_RESPONSE, NO_EXAMPLE_RESPONSE
from tests.unit_tests.testing_utils.components import CountingCallbackHandler
from tests.unit_tests.testing_utils.mocks.mock_chat_models import (
    FakeChatModelWithPresetResponses,
    FakeSimpleChatModel,
)
from tests.unit_tests.testing_utils.mocks.mock_memory import FakeMemory
from tests.unit_tests.testing_utils.mocks.mock_tool import FakeTool

from erniebot_agent.agents import FunctionalAgentWithRetrieval
from erniebot_agent.memory import AIMessage, HumanMessage
from erniebot_agent.memory.messages import FunctionCall
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
async def test_functional_agent_with_retrieval_callbacks(identity_tool):
    callback_handler = CountingCallbackHandler()
    knowledge_base_name = "test"
    access_token = "your access token"
    knowledge_base_id = 111
    with mock.patch("requests.post") as my_mock:
        search_db = BaizhongSearch(
            knowledge_base_name=knowledge_base_name,
            access_token=access_token,
            knowledge_base_id=knowledge_base_id if knowledge_base_id != "" else None,
        )
    agent = FunctionalAgentWithRetrieval(
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


@pytest.mark.asyncio
async def test_functional_agent_with_retrieval_run_retrieval(identity_tool):
    knowledge_base_name = "test"
    access_token = "your access token"
    knowledge_base_id = 111
    with mock.patch("requests.post") as my_mock:
        search_db = BaizhongSearch(
            knowledge_base_name=knowledge_base_name,
            access_token=access_token,
            knowledge_base_id=knowledge_base_id if knowledge_base_id != "" else None,
        )
    agent = FunctionalAgentWithRetrieval(
        knowledge_base=search_db, llm=FakeSimpleChatModel(), tools=[identity_tool], memory=FakeMemory()
    )

    # Test retrieval success
    with mock.patch("requests.post") as my_mock:
        my_mock.return_value = MagicMock(status_code=200, json=lambda: EXAMPLE_RESPONSE)
        response = await agent.async_run("Hello, world!")

    assert response.text == "Text response"
    # HumanMessage
    assert (
        response.chat_history[0].content
        == "检索结果:\n\n    第1个段落: 住房和城乡建设部规章城市管理执法办法\n\n    第2个段落: 城市管理执法主管部门应当定期开展执法人员的培训和考核。\n\n" \
        "检索语句: Hello, world!\n请根据以上检索结果回答检索语句的问题"
    )
    # AIMessage
    assert response.chat_history[1].content == "Text response"
    assert response.chat_history[1].search_info == {
        "results": [
            {"index": 1, "url": "", "title": "城市管理执法办法.pdf"},
            {"index": 2, "url": "", "title": "城市管理执法办法.pdf"},
        ]
    }
    # Test retrieval failed
    with mock.patch("requests.post") as my_mock:
        my_mock.return_value = MagicMock(status_code=200, json=lambda: NO_EXAMPLE_RESPONSE)
        response = await agent.async_run("Hello, world!")

    assert response.text == "Text response"
    # HumanMessage
    assert response.chat_history[0].content == "Hello, world!"
    # AIMessage
    assert response.chat_history[1].content == "Text response"


@pytest.mark.asyncio
async def test_functional_agent_with_retrieval_run_llm(identity_tool):
    output_message = AIMessage("Hello!", function_call=None)

    knowledge_base_name = "test"
    access_token = "your access token"
    knowledge_base_id = 111
    search_db = BaizhongSearch(
        knowledge_base_name=knowledge_base_name,
        access_token=access_token,
        knowledge_base_id=knowledge_base_id if knowledge_base_id != "" else None,
    )
    agent = FunctionalAgentWithRetrieval(
        knowledge_base=search_db,
        llm=FakeChatModelWithPresetResponses(responses=[output_message]),
        tools=[],
        memory=FakeMemory(),
    )
    llm_response = await agent._async_run_llm(messages=[HumanMessage("Hello, world!")])
    assert isinstance(llm_response.message, AIMessage)
    assert llm_response.message == output_message

    output_message = AIMessage(
        "",
        function_call=FunctionCall(
            name=identity_tool.tool_name, thoughts="", arguments=json.dumps({"param": "test"})
        ),
    )
    agent = FunctionalAgentWithRetrieval(
        knowledge_base=search_db,
        llm=FakeChatModelWithPresetResponses(responses=[output_message]),
        tools=[identity_tool],
        memory=FakeMemory(),
    )
    llm_response = await agent._async_run_llm(messages=[HumanMessage("Hello, world!")])
    assert isinstance(llm_response.message, AIMessage)
    assert llm_response.message == output_message


@pytest.mark.asyncio
async def test_functional_agent_with_retrieval_run_tool(identity_tool, no_input_no_output_tool):
    knowledge_base_name = "test"
    access_token = "your access token"
    knowledge_base_id = 111
    search_db = BaizhongSearch(
        knowledge_base_name=knowledge_base_name,
        access_token=access_token,
        knowledge_base_id=knowledge_base_id if knowledge_base_id != "" else None,
    )
    agent = FunctionalAgentWithRetrieval(
        knowledge_base=search_db,
        llm=FakeSimpleChatModel(),
        tools=[identity_tool, no_input_no_output_tool],
        memory=FakeMemory(),
    )

    tool_input = {"param": "test"}
    tool_response = await agent._async_run_tool(identity_tool.tool_name, json.dumps(tool_input))
    assert json.loads(tool_response.json) == tool_input

    tool_input = {}
    tool_response = await agent._async_run_tool(no_input_no_output_tool.tool_name, json.dumps(tool_input))
    assert json.loads(tool_response.json) == {}

    tool_input = {}
    with pytest.raises(RuntimeError):
        await agent._async_run_tool("some_tool_name_that_does_not_exist", json.dumps(tool_input))


@pytest.mark.asyncio
async def test_functional_agent_with_retrieval_max_steps(identity_tool):
    function_call = FunctionCall(
        name=identity_tool.tool_name,
        thoughts="",
        arguments=json.dumps({"param": "test"}),
    )
    llm = FakeChatModelWithPresetResponses(
        responses=[
            AIMessage("", function_call=function_call),
            AIMessage("", function_call=function_call),
            AIMessage("", function_call=function_call),
            AIMessage("Done.", function_call=None),
        ]
    )
    knowledge_base_name = "test"
    access_token = "your access token"
    knowledge_base_id = 111
    search_db = BaizhongSearch(
        knowledge_base_name=knowledge_base_name,
        access_token=access_token,
        knowledge_base_id=knowledge_base_id if knowledge_base_id != "" else None,
    )
    agent = FunctionalAgentWithRetrieval(
        knowledge_base=search_db,
        llm=llm,
        tools=[identity_tool],
        memory=FakeMemory(),
        max_steps=2,
    )
    with mock.patch("requests.post") as my_mock:
        my_mock.return_value = MagicMock(status_code=200, json=lambda: NO_EXAMPLE_RESPONSE)
        response = await agent.async_run("Run!")
    assert response.status == "STOPPED"
