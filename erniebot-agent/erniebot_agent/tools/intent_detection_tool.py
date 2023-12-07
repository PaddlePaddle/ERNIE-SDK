from __future__ import annotations

from typing import Optional, Type

from erniebot_agent.tools.base import Tool
from erniebot_agent.tools.schema import ToolParameterView
from prompt_utils import auto_agent_instructions
from pydantic import Field
from utils import erniebot_chat


class IntentDetectionToolInputView(ToolParameterView):
    query: str = Field(description="Chunk of text to summarize")


class IntentDetectionToolOutputView(ToolParameterView):
    document: str = Field(description="content")


class IntentDetectionTool(Tool):
    description: str = "text ranking tool"
    input_type: Type[ToolParameterView] = IntentDetectionToolInputView
    ouptut_type: Type[ToolParameterView] = IntentDetectionToolOutputView

    async def __call__(self, content: str, functions: Optional[str] = None, **kwargs):
        prompt = auto_agent_instructions()
        messages = [{"role": "user", "content": prompt + f"\ntask: {content}\n response: \n "}]
        return erniebot_chat(messages=messages, functions=functions, **kwargs)
