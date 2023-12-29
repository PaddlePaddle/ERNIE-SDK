from __future__ import annotations

import asyncio
from typing import Any, Dict, Type, List
from pydantic import Field
from erniebot_agent.tools.base import Tool

from erniebot_agent.tools.schema import ToolParameterView

from erniebot_agent.agents.functional_agent import FunctionalAgent
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.memory import WholeMemory



class AddWordInput(ToolParameterView):
    word: str = Field(description="待添加的单词")

class AddWordOutput(ToolParameterView):
    result: str = Field(description="表示是否成功将单词成功添加到词库当中")

class AddWordTool(Tool):
    description: str = "添加单词到词库当中"
    input_type: Type[ToolParameterView] = AddWordInput
    ouptut_type: Type[ToolParameterView] = AddWordOutput

    def __init__(self) -> None:
        self.word_books = {}
        super().__init__()


    async def __call__(self, word: str) -> Dict[str, Any]:
        if word in self.word_books:
            return {"result": f"<{word}>单词已经存在，无需添加"}
        self.word_books[word] = True
        words = "\n".join(list(self.word_books.keys()))
        return {"result": f"<{word}>单词已添加成功, 当前单词本中有如下单词：{words}"}


async def main():
    agent = FunctionalAgent(ERNIEBot("ernie-3.5"), tools=[AddWordTool()], memory=WholeMemory())
    result = await agent.async_run("将单词：“red”添加到单词当中")
    print(result)


asyncio.run(main())