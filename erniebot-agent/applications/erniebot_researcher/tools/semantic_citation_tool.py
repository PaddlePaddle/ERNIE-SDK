import logging
import string
from collections import OrderedDict
from typing import Optional, Type

from pydantic import Field

from erniebot_agent.chat_models.erniebot import BaseERNIEBot
from erniebot_agent.memory import HumanMessage
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
    return f"{meta_dict},根据上面的数据，生成报告的参考文献，请严格按照如下【JSON格式】的形式输出:" + json_format


import json

from .utils import write_md_to_pdf

logger = logging.getLogger(__name__)


class SemanticCitationToolInputView(ToolParameterView):
    query: str = Field(description="Chunk of text to summarize")


class SemanticCitationToolOutputView(ToolParameterView):
    document: str = Field(description="content")


class SemanticCitationTool(Tool):
    description: str = "semantic citation tool"
    input_type: Type[ToolParameterView] = SemanticCitationToolInputView
    ouptut_type: Type[ToolParameterView] = SemanticCitationToolOutputView

    def is_punctuation(self, char: str):
        """判断一个字符是否是标点符号"""
        return char in string.punctuation

    def __init__(self, llm: BaseERNIEBot, theta_min=0.4, theta_max=0.95) -> None:
        super().__init__()
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.llm = llm

    async def __call__(
        self,
        report: str,
        agent_name: str,
        report_type: str,
        dir_path: str,
        citation_faiss_research,
        meta_data: Optional[OrderedDict] = None,
        theta_min: Optional[float] = None,
        theta_max: Optional[float] = None,
        **kwargs,
    ):
        # Manually Add reference on the bottom
        meta_data_json = json.dumps(meta_data, ensure_ascii=False)
        if "参考文献" not in report:
            report += "\n\n## 参考文献 \n"
            messages = [HumanMessage(content=generate_reference(meta_data_json).replace(". ", "."))]
            response = await self.llm.chat(messages)
            result = response.content
            start_idx = result.index("{")
            end_idx = result.rindex("}")
            corrected_data = result[start_idx : end_idx + 1]
            response = json.loads(corrected_data)
            for i, item in enumerate(response["参考文献"]):
                report += f"{i+1}. {item['标题']} [链接]({item['链接']})\n"
        elif "参考文献" in report[-500:]:
            idx = report.index("参考文献")
            report = report[:idx]
            messages = [HumanMessage(content=generate_reference(meta_data_json).replace(". ", "."))]
            response = await self.llm.chat(messages)
            result = response.content
            start_idx = result.index("{")
            end_idx = result.rindex("}")
            corrected_data = result[start_idx : end_idx + 1]
            response = json.loads(corrected_data)
            for i, item in enumerate(response["参考文献"]):
                report += f"{i+1}. {item['标题']} [链接]({item['链接']})\n"
        if theta_min:
            self.theta_min = theta_min
        url_index = {}
        if meta_data:
            for index, (key, val) in enumerate(meta_data.items()):
                url_index[val] = {"name": key, "index": index + 1}
        if theta_max:
            self.theta_max = theta_max
        list_data = report.split("\n\n")
        output_text = []
        for chunk_text in list_data:
            if "参考文献" in chunk_text:
                output_text.append(chunk_text)
                break
            elif "#" in chunk_text or "摘要" in chunk_text or "关键词" in chunk_text:
                output_text.append(chunk_text)
                continue
            else:
                sentence_splits = chunk_text.split("。")
                output_sent = []
                for sentence in sentence_splits:
                    if not sentence:
                        continue
                    try:
                        query_result = citation_faiss_research.search(query=sentence, top_k=1, filters=None)
                    except Exception as e:
                        output_sent.append(sentence)
                        logger.error(f"Faiss search error: {e}")
                        continue
                    source = query_result[0]["url"]
                    if len(sentence.strip()) > 0:
                        if not self.is_punctuation(sentence[-1]):
                            sentence += "。"
                        if (
                            query_result[0]["score"] >= self.theta_min
                            and query_result[0]["score"] <= self.theta_max
                        ):
                            if (
                                len(output_sent) > 0
                                and f"<sup>[\\[{url_index[source]['index']}\\]]({source})</sup>"
                                in output_sent[-1]
                            ):
                                output_sent[-1] = output_sent[-1].replace(
                                    f"<sup>[\\[{url_index[source]['index']}\\]]({source})</sup>", ""
                                )
                            sentence += f"<sup>[\\[{url_index[source]['index']}\\]]({source})</sup>"
                    output_sent.append(sentence)
                chunk_text = "".join(output_sent)
                output_text.append(chunk_text)
        final_report = "\n\n".join(output_text)
        path = write_md_to_pdf(agent_name + "__" + report_type, dir_path, final_report)
        return final_report, path
