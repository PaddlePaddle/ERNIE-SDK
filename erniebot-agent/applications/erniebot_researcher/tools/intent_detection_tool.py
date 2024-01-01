from __future__ import annotations

import json
from typing import Optional, Type

from erniebot_agent.tools.base import Tool
from erniebot_agent.tools.schema import ToolParameterView
from pydantic import Field

from .prompt_utils import auto_agent_instructions
from .utils import erniebot_chat


class IntentDetectionToolInputView(ToolParameterView):
    query: str = Field(description="Chunk of text to summarize")


class IntentDetectionToolOutputView(ToolParameterView):
    document: str = Field(description="content")


class IntentDetectionTool(Tool):
    description: str = "query intent detection tool"
    input_type: Type[ToolParameterView] = IntentDetectionToolInputView
    ouptut_type: Type[ToolParameterView] = IntentDetectionToolOutputView

    async def __call__(self, content: str, functions: Optional[str] = None, **kwargs):
        prompt = auto_agent_instructions()
        messages = [{"role": "user", "content": prompt.format(content=content)}]
        result = erniebot_chat(messages=messages, functions=functions, **kwargs)
        # parse json object
        start_idx = result.index("{")
        end_idx = result.rindex("}")
        result = result[start_idx : end_idx + 1]
        result = json.loads(result)
        return result
