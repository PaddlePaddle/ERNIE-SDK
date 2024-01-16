"""Test ERNIE Bot API wrapper."""

from typing import Generator

import pytest
from langchain.schema import LLMResult

from erniebot_agent.extensions.langchain.llms import ErnieBot


def test_erniebot_call() -> None:
    """Test valid call."""
    llm = ErnieBot()
    output = llm("Hi, erniebot.")
    assert isinstance(output, str)


def test_erniebot_generate() -> None:
    """Test generation."""
    llm = ErnieBot()
    output = llm.generate(["Hi, erniebot."])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)


@pytest.mark.asyncio
async def test_erniebot_agenerate() -> None:
    """Test asynchronous generation."""
    llm = ErnieBot()
    output = await llm.agenerate(["Hi, erniebot."])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)


def test_erniebot_streaming_generate() -> None:
    """Test generation with streaming enabled."""
    llm = ErnieBot(streaming=True)
    output = llm.generate(["Write a joke."])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)


def test_erniebot_stream() -> None:
    """Test streaming."""
    llm = ErnieBot()
    output = llm.stream("Write a joke.")
    assert isinstance(output, Generator)
    for res in output:
        assert isinstance(res, str)


@pytest.mark.asyncio
async def test_erniebot_astream() -> None:
    """Test asynchronous streaming."""
    llm = ErnieBot()
    output = llm.astream("Write a joke.")
    async for res in output:
        assert isinstance(res, str)
