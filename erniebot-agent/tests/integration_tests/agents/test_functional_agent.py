import json

import pytest

from erniebot_agent.agents.functional_agent import FunctionalAgent
from erniebot_agent.chat_models.erniebot import ERNIEBot
from erniebot_agent.memory.whole_memory import WholeMemory
from erniebot_agent.messages import AIMessage, FunctionMessage, HumanMessage
from erniebot_agent.tools.calculator_tool import CalculatorTool

ONE_HIT_PROMPT = "1+4等于几？"
NO_HIT_PROMPT = "深圳今天天气怎么样？"


@pytest.fixture(scope="module")
def llm():
    return ERNIEBot(model="ernie-3.5")


@pytest.fixture(scope="module")
def tool():
    return CalculatorTool()


@pytest.fixture(scope="function")
def memory():
    return WholeMemory()


@pytest.mark.asyncio
async def test_functional_agent_run_one_hit(llm, tool, memory):
    agent = FunctionalAgent(llm=llm, tools=[tool], memory=memory)
    prompt = ONE_HIT_PROMPT

    response = await agent.async_run(prompt)

    messages = response.chat_history
    assert len(messages) == 4
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == prompt
    assert isinstance(messages[1], AIMessage)
    assert messages[1].function_call is not None
    assert messages[1].function_call["name"] == tool.tool_name
    assert isinstance(messages[2], FunctionMessage)
    assert messages[2].name == messages[1].function_call["name"]
    assert json.loads(messages[2].content) == {"formula_result": 5}
    assert isinstance(messages[3], AIMessage)
    assert messages[3].content == response.text

    actions = response.actions
    assert len(actions) == 1
    assert actions[0].tool_name == tool.tool_name


@pytest.mark.asyncio
async def test_functional_agent_run_no_hit(llm, tool, memory):
    agent = FunctionalAgent(llm=llm, tools=[tool], memory=memory)
    prompt = NO_HIT_PROMPT

    response = await agent.async_run(prompt)

    messages = response.chat_history
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == prompt
    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == response.text

    assert len(response.actions) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", [ONE_HIT_PROMPT, NO_HIT_PROMPT])
async def test_functional_agent_run_no_tool(llm, memory, prompt):
    agent = FunctionalAgent(llm=llm, tools=[], memory=memory)

    response = await agent.async_run(prompt)

    messages = response.chat_history
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == prompt
    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == response.text

    assert len(response.actions) == 0
