from __future__ import annotations

import json
from collections import OrderedDict
from typing import Optional, Type

from erniebot_agent.tools.base import Tool
from erniebot_agent.tools.schema import ToolParameterView
from pydantic import Field

from .prompt_utils import generate_reference, get_report_by_type
from .utils import call_function, erniebot_chat

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
        # map reduce
        research_summary = research_summary[: TOKEN_MAX_LENGTH - 600]
        report_type_func = get_report_by_type(report_type)
        final_report = call_function(
            report_type_func(question, research_summary, outline), agent_role_prompt=agent_role_prompt
        )
        # Manually Add reference on the bottom
        if "参考文献" not in final_report:
            final_report += "\n\n## 参考文献 \n"
            messages = [{"role": "user", "content": generate_reference(meta_data)}]
            response = erniebot_chat(messages)
            start_idx = response.index("{")
            end_idx = response.rindex("}")
            corrected_data = response[start_idx : end_idx + 1]
            response = json.loads(corrected_data)
            for i, item in enumerate(response["参考文献"]):
                final_report += f"{i+1}. {item['标题']} [链接]({item['链接']})\n"
        elif "参考文献" in final_report[-500:]:
            idx = final_report.index("参考文献")
            final_report = final_report[idx + 4 :]
            messages = [{"role": "user", "content": generate_reference(meta_data)}]
            response = erniebot_chat(messages)
            start_idx = response.index("{")
            end_idx = response.rindex("}")
            corrected_data = response[start_idx : end_idx + 1]
            response = json.loads(corrected_data)
            for i, item in enumerate(response["参考文献"]):
                final_report += f"{i+1}. {item['标题']} [链接]({item['链接']})\n"
        url_index = {}
        if meta_data:
            for index, (key, val) in enumerate(meta_data.items()):
                url_index[val] = {"name": key, "index": index + 1}
        return final_report, url_index
