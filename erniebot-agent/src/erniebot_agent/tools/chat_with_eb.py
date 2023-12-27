from typing import Dict, Type

from pydantic import Field

from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.memory.messages import HumanMessage
from erniebot_agent.tools.base import Tool
from erniebot_agent.tools.schema import ToolParameterView


class ChatWithEBInputView(ToolParameterView):
    query: str = Field(description="需要询问EB的具体问题")


class ChatWithEBOutputView(ToolParameterView):
    response: str = Field(description="EB回复的结果")


class ChatWithEB(Tool):
    description: str = (
        "ChatWithEB是一款根据用户的问题，向EB生成式大语言模型进行提问，并获取EB回答结果的工具。EB一般能解决知识型问答、文本创作、信息查询、信息检索等基础的文本生成和信息检索功能"
    )
    input_type: Type[ToolParameterView] = ChatWithEBInputView
    ouptut_type: Type[ToolParameterView] = ChatWithEBOutputView

    def __init__(self, llm: ERNIEBot):
        self.llm = llm

    async def __call__(self, query: str) -> Dict[str, str]:
        response = await self.llm.chat([HumanMessage(query)])
        return {"response": response.content}
