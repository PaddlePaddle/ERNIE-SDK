from __future__ import annotations

import pytest

from erniebot_agent.tools.current_time_tool import CurrentTimeTool


@pytest.fixture(scope="module")
def tool():
    return CurrentTimeTool()


def test_schema(tool):
    function_call_schema = tool.function_call_schema()
    assert function_call_schema["description"] == CurrentTimeTool.description
    assert function_call_schema["responses"]["properties"]["current_time"]["type"] == "string"


@pytest.mark.asyncio
async def test_tool(tool):
    cur_time = await tool()
    print(cur_time)
    assert len(cur_time["current_time"]) > 6
