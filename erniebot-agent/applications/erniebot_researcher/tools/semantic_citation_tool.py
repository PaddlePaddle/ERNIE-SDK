import logging
import string
from typing import Dict, List, Optional

from zhon import hanzi

from erniebot_agent.tools.base import Tool

from .utils import write_md_to_pdf

logger = logging.getLogger(__name__)


def generate_reference(meta_dict):
    json_format = """{
            "参考文献": [
                {
                "标题": "文章标题",
                "链接": "文章链接",
                }]
            }"""
    return f"{meta_dict},根据上面的数据，生成报告的参考文献，请严格按照如下【JSON格式】的形式输出:" + json_format


logger = logging.getLogger(__name__)


class SemanticCitationTool(Tool):
    description: str = "semantic citation tool"

    def is_punctuation(self, char: str):
        """判断一个字符是否是标点符号"""
        return char in string.punctuation or char in hanzi.punctuation

    def __init__(self, theta_min=0.4, theta_max=0.95, citation_num=5) -> None:
        super().__init__()
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.citation_num = citation_num
        self.recoder_cite_dict: Dict = {}
        self.recoder_cite_list: List = []
        self.recoder_cite_title: List = []

    async def add_url_sentences(self, sententces: str, citation_research):
        sentence_splits = sententces.split("。")
        output_sent = []
        for sentence in sentence_splits:
            if not sentence:
                continue
            try:
                query_result = await citation_research(query=sentence, top_k=3, filters=None)
            except Exception as e:
                output_sent.append(sentence)
                logger.error(f"Faiss search error: {e}")
                continue
            if len(sentence.strip()) > 0:
                if not self.is_punctuation(sentence[-1]) or sentence[-1] == "%":
                    sentence += "。"
            for item in query_result["documents"]:
                source = item["meta"]["url"]
                if item["score"] >= self.theta_min and item["score"] <= self.theta_max:
                    if source not in self.recoder_cite_list:
                        self.recoder_cite_title.append(item["meta"]["name"])
                        self.recoder_cite_list.append(source)
                        self.recoder_cite_dict[source] = 1
                        index = len(self.recoder_cite_list)
                        sentence += f"<sup>[\\[{index}\\]]({source})</sup>"
                        break
                    else:
                        index = self.recoder_cite_list.index(source) + 1
                        if (
                            len(output_sent) > 0
                            and f"<sup>[\\[{index}\\]]({source})</sup>" in output_sent[-1]
                        ):
                            output_sent[-1] = output_sent[-1].replace(
                                f"<sup>[\\[{index}\\]]({source})</sup>", ""
                            )
                            sentence += f"<sup>[\\[{index}\\]]({source})</sup>"
                            break
                        else:
                            if self.recoder_cite_dict[source] >= self.citation_num:
                                continue
                            else:
                                self.recoder_cite_dict[source] += 1
                                sentence += f"<sup>[\\[{index}\\]]({source})</sup>"
                                break
            output_sent.append(sentence)
        return output_sent

    async def add_url_report(self, report: str, citation_research):
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
                output_sent = await self.add_url_sentences(chunk_text, citation_research)
                chunk_text = "".join(output_sent)
                output_text.append(chunk_text)
        report = "\n\n".join(output_text)
        return report

    def add_reference_report(self, report: str):
        # Manually Add reference on the bottom
        if "参考文献" not in report:
            report += "\n\n## 参考文献 \n"
            for index in range(len(self.recoder_cite_list)):
                title = self.recoder_cite_title[index]
                url = self.recoder_cite_list[index]
                report += f"{index+1}. {title} [链接]({url})\n"
        elif "参考文献" in report[-500:]:
            idx = report.index("参考文献")
            report = report[:idx].strip()
            if report[-1] == "#":
                report += " 参考文献 \n"
            else:
                report += "\n\n## 参考文献 \n"
            for index in range(len(self.recoder_cite_list)):
                title = self.recoder_cite_title[index]
                url = self.recoder_cite_list[index]
                report += f"{index+1}. {title} [链接]({url})\n"
        return report

    async def __call__(
        self,
        report: str,
        agent_name: str,
        report_type: str,
        dir_path: str,
        citation_research,
        citation_num: Optional[int] = None,
        theta_min: Optional[float] = None,
        theta_max: Optional[float] = None,
    ):
        if theta_min:
            self.theta_min = theta_min
        if theta_max:
            self.theta_max = theta_max
        if citation_num:
            self.citation_num = citation_num
        report = await self.add_url_report(report, citation_research)
        report = self.add_reference_report(report)
        path = write_md_to_pdf(agent_name + "__" + report_type, dir_path, report)
        return report, path
