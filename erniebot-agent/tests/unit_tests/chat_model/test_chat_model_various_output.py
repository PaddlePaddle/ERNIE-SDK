from erniebot-agent.src.erniebot_agent.memory.messages import HumanMessage
from tests.unit_tests.testing_utils.mocks.mock_chat_models import (
    FakeERNIEBotWithPresetResponses,
)
import pytest

# 1. fake various output from erniebot
@pytest.fixture(scope="module")
def fake_erniebot_with_search_info():
    fake_response_with_seach_info = {}
    return FakeERNIEBotWithPresetResponses(fake_response_with_seach_info)


@pytest.fixture(scope="module")
def fake_erniebot_with_plugin_info():
    fake_plugin_info_response = {}
    return FakeERNIEBotWithPresetResponses(fake_plugin_info_response)

@pytest.fixture(scope="module")
def fake_erniebot_with_function_call():
    fake_function_call_response = {}
    return FakeERNIEBotWithPresetResponses(fake_function_call_response)

@pytest.fixture(scope="module")
def fake_erniebot_with_vanilla_output():
    fake_vanilla_message_response = {}
    return FakeERNIEBotWithPresetResponses(fake_vanilla_message_response)


# 2. tests each output independently
@pytest.mark.asyncio
async def test_erniebot_with_search_info(stream=False):
    fake_erniebot = fake_erniebot_with_search_info(stream)
    messages = [HumanMessage("今天深圳天气怎么样？")]
    response = await fake_erniebot.async_chat(messages, stream)

    assert len(response.search_infor)>0

@pytest.mark.asyncio
async def test_erniebot_with_plugin_info(stream=False):
    fake_erniebot = fake_erniebot_with_search_info(stream)
    messages = [HumanMessage("今天深圳天气怎么样？")]
    response = await fake_erniebot.async_chat(messages, stream)

    assert len(response.plugin_info)>0

@pytest.mark.asyncio
async def test_erniebot_with_function_call(stream=False):
    fake_erniebot = fake_erniebot_with_function_call(stream)
    messages = [HumanMessage("今天深圳天气怎么样？")]
    response = await fake_erniebot.async_chat(messages, stream)

    assert len(response.function_call)>0

@pytest.mark.asyncio
async def test_erniebot_with_plugin_info(stream=False):
    fake_erniebot = fake_erniebot_with_search_info(stream)
    messages = [HumanMessage("今天深圳天气怎么样？")]
    response = await fake_erniebot.async_chat(messages, stream)

    assert response.plugin_info == None
    assert response.function_call == None
    assert response.search_info == None