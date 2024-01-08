from __future__ import annotations

from typing import List

from tools.utils import JsonUtil

from erniebot_agent.chat_models.erniebot import BaseERNIEBot
from erniebot_agent.memory import HumanMessage
from erniebot_agent.tools.base import Tool


class OutlineGenerationTool(Tool, JsonUtil):
    description: str = "text outline generation tool"

    def __init__(self, llm: BaseERNIEBot) -> None:
        super().__init__()
        self.llm = llm

    async def __call__(
        self,
        queries: List[str],
        question: str,
        **kwargs,
    ):
        ques = ""
        for i, query in enumerate(queries):
            ques += f"{i+1}. {query}\n"

        messages = [
            HumanMessage(
                content=f"""{ques}，请根据上面的问题生成一个关于"{question}"的大纲，
                大纲的最后章节是参考文献章节，字数控制在300字以内,并使用json的形式输出。"""
            )
        ]
        response = await self.llm.chat(messages=messages)
        outline = response.content
        outline = self.parse_json(outline)
        return outline
