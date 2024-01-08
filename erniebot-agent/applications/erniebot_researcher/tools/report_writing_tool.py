from __future__ import annotations

import json
from builtins import dict

from erniebot_agent.chat_models.erniebot import BaseERNIEBot
from erniebot_agent.memory import HumanMessage
from erniebot_agent.prompt import PromptTemplate
from erniebot_agent.tools.base import Tool

from .utils import write_md_to_pdf


def generate_report_prompt(question, research_summary, outline=None):
    """Generates the report prompt for the given question and research summary.
    Args: question (str): The question to generate the report prompt for
            research_summary (str): The research summary to generate the report prompt for
    Returns: str: The report prompt for the given question and research summary
    """
    if isinstance(outline, dict):
        outline = json.dumps(outline, ensure_ascii=False)
    if outline is None:
        prompt = """你是任务是生成一份满足要求的报告，报告的格式必须是markdown格式，注意报告标题前面必须有'#'
        现在给你一些信息，帮助你进行报告生成任务
        信息：{{information}}
        使用上述信息，详细报告回答以下问题或主题{{question}}
        -----
        报告应专注于回答问题，结构良好，内容丰富，包括事实和数字（如果有的话），字数控制在2400字，并采用Markdown语法和APA格式。
        注意报告标题前面必须有'#'。
        请注意生成的报告第一行必须是题目，第二行必须是一级标题。题目和一级标题之间没有其它内容。
        您必须基于给定信息确定自己的明确和有效观点。不要得出一般和无意义的结论。
        在报告末尾以APA格式列出所有使用的来源URL。
        """
        report_prompt = PromptTemplate(prompt, input_variables=["information", "question"])
        strs = report_prompt.format(information=research_summary, question=question)
    else:
        outline = outline.replace('"', "'")
        # remove spaces
        outline = outline.replace(" ", "")
        prompt = """你是任务是生成一份满足要求的报告，报告的格式必须是markdown格式，注意报告标题前面必须有'#'
        现在给你一些信息，帮助你进行报告生成任务
        信息：{{information}}
        使用上述信息，根据设定好的大纲{{outline}}
        详细报告回答以下问题或主题{{question}}
        -----
        报告应专注于回答问题，结构良好，内容丰富，包括事实和数字（如果有的话），字数控制在2400字，并采用Markdown语法和APA格式。
        注意报告标题前面必须有'#'。
        请注意生成的报告第一行必须是题目，第二行必须是一级标题。题目和一级标题之间没有其它内容。
        您必须基于给定信息确定自己的明确和有效观点。不要得出一般和无意义的结论。
        在报告末尾以APA格式列出所有使用的来源URL。
        """
        report_prompt = PromptTemplate(prompt, input_variables=["information", "outline", "question"])
        strs = report_prompt.format(information=research_summary, outline=outline, question=question)
    return strs.replace(". ", ".")


def generate_resource_report_prompt(question, research_summary, **kwargs):
    """Generates the resource report prompt for the given question and research summary.

    Args:
        question (str): The question to generate the resource report prompt for.
        research_summary (str): The research summary to generate the resource report prompt for.

    Returns:
        str: The resource report prompt for the given question and research summary.
    """
    prompt = """
    {{information}}根据上述信息，为以下问题或主题生成一份参考文献推荐报告"{{question}}"。
    该报告应详细分析每个推荐的资源，解释每个来源如何有助于找到研究问题的答案。
    着重考虑每个来源的相关性、可靠性和重要性。确保报告结构良好，信息丰富，深入，并遵循Markdown语法。
    在可用时包括相关的事实、数字和数据。报告的最低长度应为1,200字。
    """
    report_prompt = PromptTemplate(prompt, input_variables=["information", "question"])
    strs = report_prompt.format(information=research_summary, question=question)
    return strs.replace(". ", ".")


def generate_outline_report_prompt(question, research_summary, **kwargs):
    """Generates the outline report prompt for the given question and research summary.
    Args: question (str): The question to generate the outline report prompt for
            research_summary (str): The research summary to generate the outline report prompt for
    Returns: str: The outline report prompt for the given question and research summary
    """
    report_prompt = """{{information}}使用上述信息，为以下问题或主题：
    "{{question}}". 生成一个Markdown语法的研究报告大纲。
    大纲应为研究报告提供一个良好的结构框架，包括主要部分、子部分和要涵盖的关键要点。
    研究报告应详细、信息丰富、深入，至少1,200字。使用适当的Markdown语法来格式化大纲，确保可读性。
    """
    Report_prompt = PromptTemplate(report_prompt, input_variables=["information", "question"])
    strs = Report_prompt.format(information=research_summary, question=question)
    return strs.replace(". ", ".")


def get_report_by_type(report_type):
    report_type_mapping = {
        "research_report": generate_report_prompt,
        "resource_report": generate_resource_report_prompt,
        "outline_report": generate_outline_report_prompt,
    }
    return report_type_mapping[report_type]


TOKEN_MAX_LENGTH = 8000


# TOKEN_MAX_LENGTH = 4800


class ReportWritingTool(Tool):
    description: str = "report writing tool"

    def __init__(self, llm: BaseERNIEBot, llm_long: BaseERNIEBot) -> None:
        super().__init__()
        self.llm = llm
        self.llm_long = llm_long

    async def __call__(
        self,
        question: str,
        research_summary: str,
        report_type: str,
        agent_role_prompt: str,
        agent_name: str,
        dir_path: str,
        outline=None,
        **kwargs,
    ):
        research_summary = research_summary[: TOKEN_MAX_LENGTH - 600]
        report_type_func = get_report_by_type(report_type)
        messages = [HumanMessage(report_type_func(question, research_summary, outline))]
        response = await self.llm_long.chat(messages, system=agent_role_prompt)
        final_report = response.content
        if final_report == "":
            raise Exception("报告生成错误")
        path = write_md_to_pdf(agent_name + "__" + report_type, dir_path, final_report)
        return final_report, path
