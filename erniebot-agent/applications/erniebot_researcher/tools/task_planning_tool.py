from __future__ import annotations

import json
import logging
from typing import Optional

from erniebot_agent.chat_models.erniebot import BaseERNIEBot
from erniebot_agent.memory import HumanMessage
from erniebot_agent.prompt import PromptTemplate
from erniebot_agent.tools.base import Tool

logger = logging.getLogger(__name__)


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


class TaskPlanningTool(Tool):
    description: str = "query task planning tool"

    def __init__(self, llm: BaseERNIEBot) -> None:
        super().__init__()
        self.llm = llm

    async def __call__(
        self,
        question: str,
        agent_role_prompt: str,
        context: Optional[str] = None,
        is_comprehensive: bool = False,
        **kwargs,
    ):
        if not context:
            messages = [HumanMessage(content=generate_search_queries_prompt(question))]
            response = await self.llm.chat(messages, system=agent_role_prompt, temperature=0.7)
            result = response.content
        else:
            try:
                if not is_comprehensive:
                    messages = [
                        HumanMessage(content=generate_search_queries_with_context(context, question))
                    ]
                    response = await self.llm.chat(messages, system=agent_role_prompt, temperature=0.7)
                    result = response.content
                else:
                    messages = [
                        HumanMessage(
                            content=generate_search_queries_with_context_comprehensive(context, question)
                        )
                    ]
                    response = await self.llm.chat(messages, system=agent_role_prompt, temperature=0.7)
                    result = response.content

                start_idx = result.index("[")
                end_idx = result.rindex("]")
                result = result[start_idx : end_idx + 1]
                plan = json.loads(result)
            except Exception as e:
                logger.error(e)
                plan = []
        return plan
