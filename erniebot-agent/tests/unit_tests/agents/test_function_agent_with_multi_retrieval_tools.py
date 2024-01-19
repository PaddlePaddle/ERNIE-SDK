import json
from unittest import mock
from unittest.mock import MagicMock

import pytest

from erniebot_agent.agents import FunctionAgentWithMultiRetrievalTools
from erniebot_agent.memory import AIMessage
from erniebot_agent.memory.messages import FunctionCall
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
async def test_function_agent_with_multi_retrieval_tools_run_retrieval_tools(identity_tool):
    callback_handler = CountingCallbackHandler()
    knowledge_base_name = "test"
    access_token = "your access token"
    knowledge_base_id = 111
    summary_db = BaizhongSearch(
        knowledge_base_name=knowledge_base_name,
        access_token=access_token,
        knowledge_base_id=knowledge_base_id if knowledge_base_id != "" else None,
    )
    summary_tool = BaizhongSearchTool(name="summary_search", db=summary_db, description="使用这个工具总结与建筑规范相关的问题")

    full_db = BaizhongSearch(
        knowledge_base_name=knowledge_base_name,
        access_token=access_token,
        knowledge_base_id=knowledge_base_id if knowledge_base_id != "" else None,
    )
    fulltext_tool = BaizhongSearchTool(
        name="fulltext_search", db=full_db, description="使用这个工具检索特定的上下文，以回答有关建筑规范具体的问题"
    )

    llm = FakeERNIEBotWithPresetResponses(
        responses=[
            AIMessage(
                "",
                function_call=FunctionCall(
                    name=summary_tool.tool_name, thoughts="", arguments=json.dumps({"query": "住房规范有哪些？"})
                ),
            ),
            AIMessage(
                "",
                function_call=FunctionCall(
                    name=fulltext_tool.tool_name,
                    thoughts="",
                    arguments=json.dumps({"query": "建筑设计规范有哪些规章？"}),
                ),
            ),
            AIMessage("Text response", function_call=None),
        ]
    )
    agent = FunctionAgentWithMultiRetrievalTools(
        knowledge_base=summary_tool,
        llm=llm,
        threshold=0.0,
        tools=[summary_tool, fulltext_tool, identity_tool],
        memory=FakeMemory(),
        callbacks=[callback_handler],
    )
    with mock.patch("requests.post") as my_mock:
        my_mock.return_value = MagicMock(status_code=200, json=lambda: EXAMPLE_RESPONSE)
        response = await agent.run("Hello, world!")
    assert response.text == "Text response"


@pytest.mark.asyncio
async def test_function_agent_with_multi_retrieval_tools_max_steps(identity_tool):
    knowledge_base_name = "test"
    access_token = "your access token"
    knowledge_base_id = 111
    summary_db = BaizhongSearch(
        knowledge_base_name=knowledge_base_name,
        access_token=access_token,
        knowledge_base_id=knowledge_base_id if knowledge_base_id != "" else None,
    )
    summary_tool = BaizhongSearchTool(name="summary_search", db=summary_db, description="使用这个工具总结与建筑规范相关的问题")

    full_db = BaizhongSearch(
        knowledge_base_name=knowledge_base_name,
        access_token=access_token,
        knowledge_base_id=knowledge_base_id if knowledge_base_id != "" else None,
    )
    fulltext_tool = BaizhongSearchTool(
        name="fulltext_search", db=full_db, description="使用这个工具检索特定的上下文，以回答有关建筑规范具体的问题"
    )

    llm = FakeERNIEBotWithPresetResponses(
        responses=[
            AIMessage(
                "",
                function_call=FunctionCall(
                    name=summary_tool.tool_name, thoughts="", arguments=json.dumps({"query": "住房规范有哪些？"})
                ),
            ),
            AIMessage(
                "",
                function_call=FunctionCall(
                    name=fulltext_tool.tool_name,
                    thoughts="",
                    arguments=json.dumps({"query": "建筑设计规范有哪些规章？"}),
                ),
            ),
            AIMessage('{"sub_query1":"具体子问题1","sub_query2":"具体子问题2"}', function_call=None),
            AIMessage("Text response", function_call=None),
        ]
    )

    agent = FunctionAgentWithMultiRetrievalTools(
        knowledge_base=summary_tool,
        llm=llm,
        threshold=0.0,
        tools=[summary_tool, fulltext_tool, identity_tool],
        memory=FakeMemory(),
        max_steps=2,
    )
    with mock.patch("requests.post") as my_mock:
        my_mock.return_value = MagicMock(status_code=200, json=lambda: EXAMPLE_RESPONSE)
        response = await agent.run("Hello, world!")

    assert response.text == "Text response"
    assert response.status == "STOPPED"
