from __future__ import annotations

import json
from typing import Optional, Type

from pydantic import Field

from erniebot_agent.prompt import PromptTemplate
from erniebot_agent.tools.base import Tool
from erniebot_agent.tools.schema import ToolParameterView

from .utils import call_function


def generate_search_queries_prompt(question):
    """Generates the search queries prompt for the given question.
    Args: question (str): The question to generate the search queries prompt for
    Returns: str: The search queries prompt for the given question
    """
    prompt = """
    写出 4 个谷歌搜索查询，以从以下内容形成客观意见： "{{question}}"
    您必须以以下格式回复一个中文字符串列表：["query 1", "query 2", "query 3", "query 4"].
    """
    queries_prompt = PromptTemplate(prompt, input_variables=["question"])
    return queries_prompt.format(question=question)


def generate_search_queries_with_context(context, question):
    """Generates the search queries prompt for the given question.
    Args: question (str): The question to generate the search queries prompt for
    Returns: str: The search queries prompt for the given question
    """
    prompt = """
    {{context}} 根据上述信息，写出 4 个搜索查询，以从以下内容形成客观意见： "{{question}}"
    您必须以以下格式回复一个中文字符串列表：["query 1", "query 2", "query 3", "query 4"].
    """
    queries_prompt = PromptTemplate(prompt, input_variables=["context", "question"])
    return queries_prompt.format(context=context, question=question)


def generate_search_queries_with_context_comprehensive(context, question):
    """Generates the search queries prompt for the given question.
    Args: question (str): The question to generate the search queries prompt for
    Returns: str: The search queries prompt for the given question
    """
    context_comprehensive = """
    你的任务是根据给出的多篇context内容，综合考虑这些context的内容，写出4个综合性搜索查询。现在多篇context为{{context}}
    你需要综合考虑上述信息，写出 4 个综合性搜索查询，以从以下内容形成客观意见： "{{question}}"
    您必须以以下格式回复一个中文字符串列表：["query 1", "query 2", "query 3", "query 4"]。
    """
    prompt = PromptTemplate(context_comprehensive, input_variables=["context", "question"])
    return prompt.format(context=str(context), question=question)


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
        is_comprehensive: bool = False,
        **kwargs,
    ):
        if not context:
            result = call_function(
                action=generate_search_queries_prompt(question),
                agent_role_prompt=agent_role_prompt,
                temperature=0.7,
            )
        else:
            try:
                if not is_comprehensive:
                    result = call_function(
                        action=generate_search_queries_with_context(context, question),
                        agent_role_prompt=agent_role_prompt,
                        temperature=0.7,
                    )
                else:
                    result = call_function(
                        action=generate_search_queries_with_context_comprehensive(context, question),
                        agent_role_prompt=agent_role_prompt,
                        temperature=0.7,
                    )
                start_idx = result.index("[")
                end_idx = result.rindex("]")
                result = result[start_idx : end_idx + 1]
                plan = json.loads(result)
            except Exception as e:
                print(e)
                plan = []
        return plan
