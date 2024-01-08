import json

import pytest

from erniebot_agent.agents import FunctionAgent
from erniebot_agent.memory import AIMessage, HumanMessage
from erniebot_agent.memory.messages import FunctionCall
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
async def test_function_agent_callbacks(identity_tool):
    callback_handler = CountingCallbackHandler()
    agent = FunctionAgent(
        llm=FakeERNIEBotWithPresetResponses([AIMessage("Hello!", function_call=None)] * 2),
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

    await agent.run("Hello, world!")
    assert callback_handler.run_starts == 1
    assert callback_handler.run_ends == 1


@pytest.mark.asyncio
async def test_function_agent_load_unload_tools(identity_tool, no_input_no_output_tool):
    tool1 = identity_tool
    tool2 = no_input_no_output_tool

    agent = FunctionAgent(
        llm=FakeERNIEBotWithPresetResponses([]),
        tools=[tool1],
        memory=FakeMemory(),
    )

    agent.load_tool(tool2)
    with pytest.raises(ValueError):
        agent.load_tool(tool1)

    agent.unload_tool(tool1)
    with pytest.raises(ValueError):
        agent.unload_tool(tool1)


@pytest.mark.asyncio
async def test_function_agent_run_llm_return_text():
    output_message = AIMessage("Hello!", function_call=None)
    agent = FunctionAgent(
        llm=FakeERNIEBotWithPresetResponses(responses=[output_message]),
        tools=[],
        memory=FakeMemory(),
    )

    llm_response = await agent.run_llm(messages=[HumanMessage("Hello, world!")])

    assert isinstance(llm_response.message, AIMessage)
    assert llm_response.message == output_message


@pytest.mark.asyncio
async def test_function_agent_run_llm_return_function_call(identity_tool):
    output_message = AIMessage(
        "",
        function_call=FunctionCall(
            name=identity_tool.tool_name, thoughts="", arguments=json.dumps({"param": "test"})
        ),
    )
    agent = FunctionAgent(
        llm=FakeERNIEBotWithPresetResponses(responses=[output_message]),
        tools=[identity_tool],
        memory=FakeMemory(),
    )

    llm_response = await agent.run_llm(messages=[HumanMessage("Hello, world!")])

    assert isinstance(llm_response.message, AIMessage)
    assert llm_response.message == output_message


@pytest.mark.asyncio
async def test_function_agent_run_tool(identity_tool, no_input_no_output_tool):
    agent = FunctionAgent(
        llm=FakeERNIEBotWithPresetResponses([]),
        tools=[identity_tool, no_input_no_output_tool],
        memory=FakeMemory(),
    )

    tool_input = {"param": "test"}
    tool_response = await agent.run_tool(identity_tool.tool_name, json.dumps(tool_input))
    assert json.loads(tool_response.json) == tool_input

    tool_input = {}
    tool_response = await agent.run_tool(no_input_no_output_tool.tool_name, json.dumps(tool_input))
    assert json.loads(tool_response.json) == {}

    tool_input = {}
    with pytest.raises(ValueError):
        await agent.run_tool("some_tool_name_that_does_not_exist", json.dumps(tool_input))


@pytest.mark.asyncio
async def test_function_agent_memory(identity_tool):
    input_text = "Run!"
    output_text = "Done."

    function_call = FunctionCall(
        name=identity_tool.tool_name,
        thoughts="",
        arguments=json.dumps({"param": "test"}),
    )
    llm = FakeERNIEBotWithPresetResponses(
        responses=[
            AIMessage("", function_call=function_call),
            AIMessage("", function_call=function_call),
            AIMessage("", function_call=function_call),
            AIMessage(output_text, function_call=None),
            AIMessage(output_text, function_call=None),
            AIMessage("This message should not be remembered.", function_call=None),
            AIMessage("This message should not be remembered, either.", function_call=None),
        ]
    )
    agent = FunctionAgent(
        llm=llm,
        tools=[identity_tool],
        memory=FakeMemory(),
    )

    await agent.run(input_text)
    messages_in_memory = agent.memory.get_messages()
    assert len(messages_in_memory) == 2
    assert isinstance(messages_in_memory[0], HumanMessage)
    assert messages_in_memory[0].content == input_text
    assert isinstance(messages_in_memory[1], AIMessage)
    assert messages_in_memory[1].content == output_text

    await agent.run(input_text)
    assert len(agent.memory.get_messages()) == 2 + 2
    agent.reset_memory()
    assert len(agent.memory.get_messages()) == 0


@pytest.mark.asyncio
async def test_function_agent_max_steps(identity_tool):
    function_call = FunctionCall(
        name=identity_tool.tool_name,
        thoughts="",
        arguments=json.dumps({"param": "test"}),
    )
    llm = FakeERNIEBotWithPresetResponses(
        responses=[
            AIMessage("", function_call=function_call),
            AIMessage("", function_call=function_call),
            AIMessage("", function_call=function_call),
            AIMessage("Done.", function_call=None),
        ]
    )
    agent = FunctionAgent(
        llm=llm,
        tools=[identity_tool],
        memory=FakeMemory(),
        max_steps=2,
    )

    response = await agent.run("Run!")

    assert response.status == "STOPPED"


@pytest.mark.asyncio
async def test_function_agent_snapshot(identity_tool):
    function_call = FunctionCall(
        name=identity_tool.tool_name,
        thoughts="",
        arguments=json.dumps({"param": "test"}),
    )
    llm = FakeERNIEBotWithPresetResponses(
        responses=[
            AIMessage("", function_call=function_call),
            AIMessage("", function_call=function_call),
            AIMessage("Hello", function_call=None),
            AIMessage("World", function_call=None),
        ]
    )
    agent = FunctionAgent(
        llm=llm,
        tools=[identity_tool],
        memory=FakeMemory(),
    )

    first_run_response = await agent.run("Run!")
    snapshot = agent.snapshots.pop()
    agent.restore_from_snapshot(snapshot)
    second_run_response = await agent.run("Run!")
    assert second_run_response.chat_history[:-1] == first_run_response.chat_history[:-1]
    assert second_run_response.steps == first_run_response.steps
