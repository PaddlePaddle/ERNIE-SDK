import json
from typing import Any, Dict, Type

import pytest
from pydantic import Field

from erniebot_agent.agents import FunctionAgent
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.tools.base import Tool
from erniebot_agent.tools.schema import ToolParameterView

PROMPT = "烟台这个城市怎么样？"


@pytest.fixture(scope="module")
def tool():
    # mimic
    class WeatherInput(ToolParameterView):
        location: str = Field(description="省，市名，例如：河北省")
        unit: str = Field(None, description="温度单位，例如：摄氏度")

    class WeatherOutput(ToolParameterView):
        temperature: int = Field(description="当前温度")
        weather_condition: str = Field(description="当前天气状况，例如：晴，多云，雨等")
        humidity: int = Field(None, description="当前湿度")
        wind_speed: int = Field(None, description="当前风速")

    class WeatherTool(Tool):
        description: str = "获得指定地点的天气"
        input_type: Type[ToolParameterView] = WeatherInput
        ouptut_type: Type[ToolParameterView] = WeatherOutput

        async def __call__(self, location: str, unit: str = "摄氏度") -> Dict[str, Any]:
            if location == "烟台":
                return {
                    "temperature": 20,
                    "weather_condition": "晴",
                    "humidity": 80,
                    "wind_speed": 10,
                }
            else:
                return {
                    "temperature": 25,
                    "weather_condition": "多云",
                    "humidity": 90,
                    "wind_speed": 20,
                }

    return WeatherTool()


@pytest.fixture(scope="module")
def llm():
    return ERNIEBot(model="ernie-3.5")

@pytest.mark.asyncio
async def test_tool_choice_not_exist(llm, tool):
    with pytest.raises(RuntimeError) as exc_info:
        FunctionAgent(llm=llm, first_tools=[tool], tools=[])
    assert str(exc_info.value) == "The first tool must be in the tools list."

@pytest.mark.asyncio
async def test_function_agent_run_tool_choice(llm, tool):
    agent = FunctionAgent(llm=llm, tools=[tool], first_tools=[tool])
    prompt = PROMPT

    response = await agent.run(prompt)

    assert len(response.steps) > 0
    assert response.steps[0].info["tool_name"] == tool.tool_name
    assert response.steps[0].info["tool_args"] == '{"location":"烟台"}'
    assert len(response.chat_history) > 2
    assert hasattr(response.chat_history[1], "function_call")
    assert response.chat_history[2].role == "function"
    assert json.loads(response.chat_history[2].content)["temperature"] == 20


@pytest.mark.asyncio
async def test_function_agent_run_no_tool_choice(llm, tool):
    agent = FunctionAgent(llm=llm, tools=[tool])
    prompt = PROMPT

    response = await agent.run(prompt)

    assert len(response.steps) == 0
    assert len(response.chat_history) == 2
    assert response.chat_history[1].function_call is None
