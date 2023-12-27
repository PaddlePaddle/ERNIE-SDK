from unittest import mock

import pytest

from erniebot_agent.agents import Agent
from erniebot_agent.agents.callback.callback_manager import CallbackManager
from erniebot_agent.agents.schema import AgentResponse
from erniebot_agent.chat_models.base import ChatModel
from erniebot_agent.memory import AIMessage
from erniebot_agent.tools import Tool
from tests.unit_tests.testing_utils.components import CountingCallbackHandler


@pytest.mark.asyncio
async def test_callback_manager_hit():
    def _assert_all_counts(handler):
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

    handler = CountingCallbackHandler()
    callback_manager = CallbackManager(handlers=[handler])

    await callback_manager.on_run_start(agent, "")
    assert handler.run_starts == 1

    await callback_manager.on_llm_start(agent, llm, [])
    assert handler.llm_starts == 1

    await callback_manager.on_llm_end(
        agent,
        llm,
        AIMessage(content="", function_call=None, token_usage={"prompt_tokens": 0, "completion_tokens": 0}),
    )
    assert handler.llm_ends == 1

    await callback_manager.on_llm_error(agent, llm, Exception())
    assert handler.llm_errors == 1

    await callback_manager.on_tool_start(agent, tool, "{}")
    assert handler.tool_starts == 1

    await callback_manager.on_tool_end(agent, tool, "{}")
    assert handler.tool_ends == 1

    await callback_manager.on_tool_error(agent, tool, Exception())
    assert handler.tool_errors == 1

    await callback_manager.on_run_end(
        agent, AgentResponse(text="", chat_history=[], steps=[], status="FINISHED")
    )
    assert handler.run_ends == 1

    _assert_all_counts(handler)


@pytest.mark.asyncio
async def test_callback_manager_add_remove_handlers():
    handler1 = CountingCallbackHandler()
    handler2 = CountingCallbackHandler()
    callback_manager = CallbackManager(handlers=[handler1])

    assert len(callback_manager.handlers) == 1

    callback_manager.remove_handler(handler1)
    assert len(callback_manager.handlers) == 0

    callback_manager.add_handler(handler1)
    assert len(callback_manager.handlers) == 1

    callback_manager.add_handler(handler2)
    assert len(callback_manager.handlers) == 2

    callback_manager.add_handler(handler1)
    assert len(callback_manager.handlers) == 3

    callback_manager.remove_handler(handler1)
    assert len(callback_manager.handlers) == 2

    callback_manager.remove_all_handlers()
    assert len(callback_manager.handlers) == 0
