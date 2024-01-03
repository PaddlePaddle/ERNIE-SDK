from __future__ import annotations

import json
from collections import OrderedDict
from typing import Optional, Type

from pydantic import Field

from erniebot_agent.chat_models.erniebot import BaseERNIEBot
from erniebot_agent.memory import HumanMessage
from erniebot_agent.prompt import PromptTemplate
from erniebot_agent.tools.base import Tool
from erniebot_agent.tools.schema import ToolParameterView


def generate_reference(meta_dict):
    json_format = """{
            "参考文献": [
                {
                "标题": "文章标题",
                "链接": "文章链接",
                }]
            }"""
    return (
        f""""{meta_dict},根据上面的数据，生成报告的参考文献，按照如下json的形式输出:
            """
        + json_format
    )


def generate_report_prompt(question, research_summary, outline=None):
    """Generates the report prompt for the given question and research summary.
    Args: question (str): The question to generate the report prompt for
            research_summary (str): The research summary to generate the report prompt for
    Returns: str: The report prompt for the given question and research summary
    """
    if outline is None:
        prompt = """你是任务是生成一份满足要求的报告，报告的格式必须是markdown格式，注意报告标题前面必须有'#'
        现在给你一些信息，帮助你进行报告生成任务
        信息：{{information}}
        使用上述信息，详细报告回答以下问题或主题{{question}}
        -----
        报告应专注于回答问题，结构良好，内容丰富，包括事实和数字（如果有的话），字数控制在3000字，并采用Markdown语法和APA格式。
        注意报告标题前面必须有'#'
        您必须基于给定信息确定自己的明确和有效观点。不要得出一般和无意义的结论。
        在报告末尾以APA格式列出所有使用的来源URL。
        """
        report_prompt = PromptTemplate(prompt, input_variables=["information", "question"])
        strs = report_prompt.format(information=research_summary, question=question)
    else:
        outline = outline.replace('"', "'")
        prompt = """你是任务是生成一份满足要求的报告，报告的格式必须是markdown格式，注意报告标题前面必须有'#'
        现在给你一些信息，帮助你进行报告生成任务
        信息：{{information}}
        使用上述信息，根据设定好的大纲{{outline}}
        详细报告回答以下问题或主题{{question}}
        -----
        报告应专注于回答问题，结构良好，内容丰富，包括事实和数字（如果有的话），字数控制在3000字，并采用Markdown语法和APA格式。
        注意报告标题前面必须有'#'
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
class ReportWritingToolInputView(ToolParameterView):
    query: str = Field(description="Chunk of text to summarize")


class ReportWritingToolOutputView(ToolParameterView):
    document: str = Field(description="content")


class ReportWritingTool(Tool):
    description: str = "report writing tool"
    input_type: Type[ToolParameterView] = ReportWritingToolInputView
    ouptut_type: Type[ToolParameterView] = ReportWritingToolOutputView

    def __init__(self, llm: BaseERNIEBot) -> None:
        super().__init__()
        self.llm = llm

    async def __call__(
        self,
        question: str,
        research_summary: str,
        report_type: str,
        agent_role_prompt: str,
        meta_data: Optional[OrderedDict] = None,
        outline=None,
        **kwargs,
    ):
        research_summary = research_summary[: TOKEN_MAX_LENGTH - 600]
        report_type_func = get_report_by_type(report_type)
        messages = [HumanMessage(report_type_func(question, research_summary, outline))]
        response = await self.llm.chat(messages, system=agent_role_prompt)
        final_report = response.content
        if final_report == "":
            raise Exception("报告生成错误")
        # Manually Add reference on the bottom
        if "参考文献" not in final_report:
            final_report += "\n\n## 参考文献 \n"
            messages = [HumanMessage(content=generate_reference(meta_data).replace(". ", "."))]
            response = await self.llm.chat(messages)
            result = response.content
            start_idx = result.index("{")
            end_idx = result.rindex("}")
            corrected_data = result[start_idx : end_idx + 1]
            response = json.loads(corrected_data)
            for i, item in enumerate(response["参考文献"]):
                final_report += f"{i+1}. {item['标题']} [链接]({item['链接']})\n"
        elif "参考文献" in final_report[-500:]:
            idx = final_report.index("参考文献")
            final_report = final_report[idx + 4 :]
            messages = [HumanMessage(content=generate_reference(meta_data))]
            response = await self.llm.chat(messages)
            result = response.content
            start_idx = result.index("{")
            end_idx = result.rindex("}")
            corrected_data = result[start_idx : end_idx + 1]
            response = json.loads(corrected_data)
            for i, item in enumerate(response["参考文献"]):
                final_report += f"{i+1}. {item['标题']} [链接]({item['链接']})\n"
        url_index = {}
        if meta_data:
            for index, (key, val) in enumerate(meta_data.items()):
                url_index[val] = {"name": key, "index": index + 1}
        # final_report=postprocess(final_report)
        return final_report, url_index
