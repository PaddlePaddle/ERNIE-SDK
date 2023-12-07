from __future__ import annotations

import json
from typing import Optional, Type

from erniebot_agent.tools.base import Tool
from erniebot_agent.tools.schema import ToolParameterView
from pydantic import Field

from .prompt_utils import (
    generate_search_queries_prompt,
    generate_search_queries_with_context,
)
from .utils import call_function


class TaskPlanningToolInputView(ToolParameterView):
    query: str = Field(description="Chunk of text to summarize")


class TaskPlanningToolOutputView(ToolParameterView):
    document: str = Field(description="content")


class TaskPlanningTool(Tool):
    description: str = "query task planning tool"
    input_type: Type[ToolParameterView] = TaskPlanningToolInputView
    ouptut_type: Type[ToolParameterView] = TaskPlanningToolOutputView

    async def __call__(
        self,
        question: str,
        agent_role_prompt: str,
        context: Optional[str] = None,
        model: str = "ernie-bot-8k",
        **kwargs,
    ):
        if context is None:
            result = call_function(
                action=generate_search_queries_prompt(question),
                agent_role_prompt=agent_role_prompt,
                temperature=0.7,
            )
        else:
            for i in range(3):
                try:
                    result = call_function(
                        action=generate_search_queries_with_context(context, question),
                        agent_role_prompt=agent_role_prompt,
                        temperature=0.7,
                    )
                    start_idx = result.index("[")
                    end_idx = result.rindex("]")
                    result = result[start_idx : end_idx + 1]
                    break
                except Exception as e:
                    print(e)
                    continue

        return json.loads(result)
