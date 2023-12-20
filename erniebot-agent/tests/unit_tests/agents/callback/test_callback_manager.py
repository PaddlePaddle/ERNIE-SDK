from unittest import mock

import pytest
from tests.unit_tests.testing_utils.components import CountingCallbackHandler

from erniebot_agent.agents.base import Agent
from erniebot_agent.agents.callback.callback_manager import CallbackManager
from erniebot_agent.agents.schema import AgentResponse
from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.messages import AIMessage
from erniebot_agent.tools.base import Tool


@pytest.mark.asyncio
async def test_callback_manager_hit():
    def _assert_num_calls(handler):
        assert handler.run_starts == 1
        assert handler.llm_starts == 1
        assert handler.llm_ends == 1
        assert handler.llm_errors == 1
        assert handler.tool_starts == 1
        assert handler.tool_ends == 1
        assert handler.tool_errors == 1
        assert handler.run_ends == 1

    agent = mock.Mock(spec=Agent)
    llm = mock.Mock(spec=ChatModel)
    tool = mock.Mock(spec=Tool)

    handler1 = CountingCallbackHandler()
    handler2 = CountingCallbackHandler()
    callback_manager = CallbackManager(handlers=[handler1, handler2])

    await callback_manager.on_run_start(agent, "")
    await callback_manager.on_llm_start(agent, llm, [])
    await callback_manager.on_llm_end(
        agent,
        llm,
        AIMessage(content="", function_call=None, token_usage={"prompt_tokens": 0, "completion_tokens": 0}),
    )
    await callback_manager.on_llm_error(agent, llm, Exception())
    await callback_manager.on_tool_start(agent, tool, "{}")
    await callback_manager.on_tool_end(agent, tool, "{}")
    await callback_manager.on_tool_error(agent, tool, Exception())
    await callback_manager.on_run_end(
        agent, AgentResponse(text="", chat_history=[], steps=[], files=[], status="FINISHED")
    )

    _assert_num_calls(handler1)
    _assert_num_calls(handler2)


@pytest.mark.asyncio
async def test_callback_manager_add_remove_handlers():
    handler1 = CountingCallbackHandler()
    handler2 = CountingCallbackHandler()
    callback_manager = CallbackManager(handlers=[handler1])
    assert len(callback_manager.handlers) == 1
    with pytest.raises(RuntimeError):
        callback_manager.add_handler(handler1)
    callback_manager.remove_handler(handler1)
    assert len(callback_manager.handlers) == 0
    callback_manager.add_handler(handler1)
    assert len(callback_manager.handlers) == 1
    callback_manager.add_handler(handler2)
    assert len(callback_manager.handlers) == 2
    callback_manager.remove_all_handlers()
    assert len(callback_manager.handlers) == 0
