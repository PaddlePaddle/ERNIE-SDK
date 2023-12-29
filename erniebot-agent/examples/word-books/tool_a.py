from __future__ import annotations

import asyncio
from typing import Any, Dict, Type, List
from pydantic import Field
from erniebot_agent.tools.base import Tool

from erniebot_agent.tools.schema import ToolParameterView

from erniebot_agent.agents.function_agent import FunctionAgent
from erniebot_agent.tools.current_time_tool import CurrentTimeTool
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.memory import WholeMemory



async def main():
    agent = FunctionAgent(ERNIEBot("ernie-3.5"), tools=[CurrentTimeTool()], memory=WholeMemory())
    result = await agent.run("现在是什么时候")
    print(result)


asyncio.run(main())