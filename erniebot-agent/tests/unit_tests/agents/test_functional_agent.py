import pytest
from erniebot_agent.memory.whole_memory import WholeMemory
from erniebot_agent.tools.calculator_tool import CalculatorTool

from tests.unit_tests.testing_utils.mock_erniebot import MockErnieBot


@pytest.fixture(scope="module")
def llm():
    return MockErnieBot(model="ernie-bot")


@pytest.fixture(scope="module")
def tool():
    return CalculatorTool()


@pytest.fixture(scope="function")
def memory():
    return WholeMemory()


@pytest.mark.asyncio
async def test_functional_agent_callbacks():
    pass


@pytest.mark.asyncio
async def test_functional_agent_load_unload_tools():
    pass


@pytest.mark.asyncio
async def test_functional_agent_run_tool():
    pass


@pytest.mark.asyncio
async def test_functional_agent_run_llm():
    pass


@pytest.mark.asyncio
async def test_functional_agent_memory():
    pass


@pytest.mark.asyncio
async def test_functional_agent_max_steps():
    pass
