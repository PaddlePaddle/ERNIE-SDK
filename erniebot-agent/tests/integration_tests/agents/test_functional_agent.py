import json
import logging

import pytest
import urllib3
from erniebot_agent.agents.functional_agent import FunctionalAgent
from erniebot_agent.chat_models.erniebot import ERNIEBot
from erniebot_agent.memory.whole_memory import WholeMemory
from erniebot_agent.messages import AIMessage, FunctionMessage, HumanMessage
from erniebot_agent.tools.calculator_tool import CalculatorTool

logging.basicConfig(level="DEBUG", format="%(message)s")


ONE_HIT_PROMPT = "1+4等于几？"
NO_HIT_PROMPT = "深圳今天天气怎么样？"


@pytest.fixture(scope="module")
def llm():
    return ERNIEBot(
        model="ernie-bot", api_type="aistudio", access_token="d86186382de8cceb4512efbd774b74ea72f3a9f5"
    )


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
    assert messages[3].content == response.content

    actions = response.actions
    assert len(actions) == 1
    assert actions[0].tool_name == tool.tool_name

    logging.info("****完成第一次测试****")


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
    assert messages[1].content == response.content

    assert len(response.actions) == 0

    logging.info("****完成第二次测试****")


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
    assert messages[1].content == response.content

    assert len(response.actions) == 0

    logging.info("****完成第三次测试****")
