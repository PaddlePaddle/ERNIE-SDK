import logging
import string
from typing import Optional

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
        return char in string.punctuation

    def __init__(self, theta_min=0.4, theta_max=0.95) -> None:
        super().__init__()
        self.theta_min = theta_min
        self.theta_max = theta_max

    async def __call__(
        self,
        report: str,
        agent_name: str,
        report_type: str,
        dir_path: str,
        citation_faiss_research,
        citation_num: int = 5,
        theta_min: Optional[float] = None,
        theta_max: Optional[float] = None,
        **kwargs,
    ):
        if theta_min:
            self.theta_min = theta_min
        if theta_max:
            self.theta_max = theta_max
        list_data = report.split("\n\n")
        output_text = []
        recoder_cite_list = []
        recoder_cite_title = []
        recoder_cite_dict = {}
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
                        query_result = citation_faiss_research.search(query=sentence, top_k=3, filters=None)
                    except Exception as e:
                        output_sent.append(sentence)
                        logger.error(f"Faiss search error: {e}")
                        continue
                    if len(sentence.strip()) > 0:
                        if not self.is_punctuation(sentence[-1]):
                            sentence += "。"
                    for item in query_result:
                        source = item["url"]
                        if item["score"] >= self.theta_min and item["score"] <= self.theta_max:
                            if source not in recoder_cite_list:
                                recoder_cite_title.append(item["title"])
                                recoder_cite_list.append(source)
                                recoder_cite_dict[source] = 1
                                index = len(recoder_cite_list)
                                sentence += f"<sup>[\\[{index}\\]]({source})</sup>"
                                break
                            else:
                                index = recoder_cite_list.index(source) + 1
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
                                    if recoder_cite_dict[source] >= citation_num:
                                        continue
                                    else:
                                        recoder_cite_dict[source] += 1
                                        sentence += f"<sup>[\\[{index}\\]]({source})</sup>"
                                        break
                    output_sent.append(sentence)
                chunk_text = "".join(output_sent)
                output_text.append(chunk_text)
        report = "\n\n".join(output_text)
        # Manually Add reference on the bottom
        if "参考文献" not in report:
            report += "\n\n## 参考文献 \n"
            for index in range(len(recoder_cite_list)):
                title = recoder_cite_title[index]
                url = recoder_cite_list[index]
                report += f"{index+1}. {title} [链接]({url})\n"
        elif "参考文献" in report[-500:]:
            idx = report.index("参考文献")
            report = report[:idx]
            for index in range(len(recoder_cite_list)):
                title = recoder_cite_title[index]
                url = recoder_cite_list[index]
                report += f"{index+1}. {title} [链接]({url})\n"
        path = write_md_to_pdf(agent_name + "__" + report_type, dir_path, report)
        return report, path
