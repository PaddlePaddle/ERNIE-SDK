import pytest
from erniebot_agent.agents.functional_agent import FunctionalAgent
from erniebot_agent.chat_models.erniebot import ERNIEBot
from erniebot_agent.memory.whole_memory import WholeMemory
from erniebot_agent.messages import AIMessage, FunctionMessage, HumanMessage
from erniebot_agent.tools.base import CalculatorTool

ONE_HIT_PROMPT = "1+4等于几？"
NO_HIT_PROMPT = "深圳今天天气怎么样？"


@pytest.fixture(scope="module")
def llm():
    return ERNIEBot(model="ernie-bot")


@pytest.fixture(scope="module")
def tool():
    return CalculatorTool()


@pytest.fixture(scope="function")
def memory():
    return WholeMemory()


@pytest.mark.asyncio
async def test_functional_agent_run_one_hit(llm, tool, memory):
    agent = FunctionalAgent(llm=llm, tools=[tool], memory=memory, run_memory=WholeMemory())
    prompt = ONE_HIT_PROMPT

    response = await agent.async_run(prompt)
    interm_messages = response.intermediate_messages
    assert len(interm_messages) == 4
    assert isinstance(interm_messages[0], HumanMessage)
    assert interm_messages[0].content == prompt
    assert isinstance(interm_messages[1], AIMessage)
    assert interm_messages[1].function_call is not None
    assert interm_messages[1].function_call["name"] == tool.tool_name
    assert isinstance(interm_messages[2], FunctionMessage)
    assert interm_messages[2].name == interm_messages[1].function_call["name"]
    assert interm_messages[2].content == "5"
    assert isinstance(interm_messages[3], AIMessage)
    assert interm_messages[3].content == response.content


@pytest.mark.asyncio
async def test_functional_agent_run_no_hit(llm, tool, memory):
    agent = FunctionalAgent(llm=llm, tools=[tool], memory=memory, run_memory=WholeMemory())
    prompt = NO_HIT_PROMPT

    response = await agent.async_run(prompt)
    interm_messages = response.intermediate_messages
    assert len(interm_messages) == 2
    assert isinstance(interm_messages[0], HumanMessage)
    assert interm_messages[0].content == prompt
    assert isinstance(interm_messages[1], AIMessage)
    assert interm_messages[1].content == response.content


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", [ONE_HIT_PROMPT, NO_HIT_PROMPT])
async def test_functional_agent_run_no_tool(llm, memory, prompt):
    agent = FunctionalAgent(llm=llm, tools=[], memory=memory, run_memory=WholeMemory())

    response = await agent.async_run(prompt)
    interm_messages = response.intermediate_messages
    assert len(interm_messages) == 2
    assert isinstance(interm_messages[0], HumanMessage)
    assert interm_messages[0].content == prompt
    assert isinstance(interm_messages[1], AIMessage)
    assert interm_messages[1].content == response.content
