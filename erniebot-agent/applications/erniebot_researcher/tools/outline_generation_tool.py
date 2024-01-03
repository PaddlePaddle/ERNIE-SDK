from __future__ import annotations

from typing import List, Type

from pydantic import Field

from erniebot_agent.tools.base import Tool
from erniebot_agent.tools.schema import ToolParameterView
from erniebot_agent.memory import HumanMessage
from .utils import erniebot_chat


class OutlineGenerationToolInputView(ToolParameterView):
    query: str = Field(description="Chunk of text to summarize")


class OutlineGenerationToolOutputView(ToolParameterView):
    document: str = Field(description="content")


class OutlineGenerationTool(Tool):
    description: str = "text outline generation tool"
    input_type: Type[ToolParameterView] = OutlineGenerationToolInputView
    ouptut_type: Type[ToolParameterView] = OutlineGenerationToolOutputView

    def __init__(self, llm: BaseERNIEBot)-> None:
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

        messages = [HumanMessage(content=f"""{ques}，请根据上面的问题生成一个关于"{question}"的大纲，
                大纲的最后章节是参考文献章节，字数控制在300字以内,并使用json的形式输出。""")]
        response = await self.llm.chat(messages=messages)
        outline = response.content
        start_idx = outline.index("{")
        end_idx = outline.rindex("}")
        outline = outline[start_idx : end_idx + 1]
        return outline
