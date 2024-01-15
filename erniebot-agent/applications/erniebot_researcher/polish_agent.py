import logging
import time
from typing import Any, List, Optional

from tools.semantic_citation_tool import SemanticCitationTool
from tools.utils import JsonUtil, ReportCallbackHandler, add_citation, write_md_to_pdf

from erniebot_agent.chat_models.erniebot import BaseERNIEBot
from erniebot_agent.memory import HumanMessage, Message, SystemMessage
from erniebot_agent.prompt import PromptTemplate

logger = logging.getLogger(__name__)
TOKEN_MAX_LENGTH = 4200


class PolishAgent(JsonUtil):
    DEFAULT_SYSTEM_MESSAGE = "你是一个报告润色助手，你的主要工作是报告进行内容上的润色"

    template_abstract = """
        请你总结报告并给出报告的摘要和关键词，摘要在100-200字之间，关键词不超过5个词。
        你需要输出一个json形式的字符串，内容为{"abstract":...,"keywords":...}。
        现在给你报告的内容：
        {{report}}"""

    template_polish = """你的任务是扩写和润色相关内容，
        你需要把相关内容扩写到300-400字之间，扩写的内容必须与给出的内容相关。
        下面给出内容:
        {{content}}
        扩写并润色内容为:"""

    def __init__(
        self,
        name: str,
        llm: BaseERNIEBot,
        llm_long: BaseERNIEBot,
        citation_tool: SemanticCitationTool,
        embeddings: Any,
        citation_index_name: str,
        dir_path: str,
        report_type: str,
        system_message: Optional[SystemMessage] = None,
        callbacks=None,
    ):
        self.name = name
        self.llm = llm
        self.llm_long = llm_long
        self.report_type = report_type
        self.dir_path = dir_path
        self.embeddings = embeddings
        self.citation_tool = citation_tool
        self.citation_index_name = citation_index_name
        self.system_message = (
            system_message.content if system_message is not None else self.DEFAULT_SYSTEM_MESSAGE
        )
        self.prompt_template_abstract = PromptTemplate(
            template=self.template_abstract, input_variables=["report"]
        )
        self.prompt_template_polish = PromptTemplate(
            template=self.template_polish, input_variables=["content"]
        )
        if callbacks is None:
            self._callback_manager = ReportCallbackHandler()
        else:
            self._callback_manager = callbacks

    async def run(self, report: str, summarize=None):
        await self._callback_manager.on_run_start(agent=self, prompt=report)
        agent_resp = await self._run(report, summarize)
        await self._callback_manager.on_run_end(agent=self, response=agent_resp)
        return agent_resp

    async def add_abstract(self, report: str):
        while True:
            try:
                content = self.prompt_template_abstract.format(report=report)
                messages: List[Message] = [HumanMessage(content)]
                if len(content) > TOKEN_MAX_LENGTH:
                    reponse = await self.llm_long.chat(messages)
                else:
                    reponse = await self.llm.chat(messages)
                res = reponse.content
                abstract_json = self.parse_json(res)
                abstract = abstract_json["abstract"]
                key = abstract_json["keywords"]
                if type(key) is list:
                    key = "，".join(key)
                return abstract, key
            except Exception as e:
                await self._callback_manager.on_llm_error(self, self.llm, e)
                continue

    async def polish_paragraph(self, report: str, abstract: str, key: str):
        report_list = [item for item in report.split("\n\n") if item.strip() != ""]
        if "#" in report_list[0]:
            paragraphs = [report_list[0]]
            if "##" in report_list[1]:
                paragraphs.append("**摘要** " + abstract)
                paragraphs.append("**关键词** " + key)
            content = ""
            for item in report_list[1:]:
                # paragraphs
                if "#" not in item:
                    content += item + "\n"
                # Title
                else:
                    # Not to polish
                    if len(content) > 300:
                        paragraphs.append(content)
                    # Polishing
                    elif len(content) > 0:
                        content = self.prompt_template_polish.format(content=content)
                        messages: List[Message] = [HumanMessage(content)]
                        try:
                            reponse = await self.llm.chat(messages)
                        except Exception as e:
                            await self._callback_manager.on_llm_error(self, self.llm, e)
                            time.sleep(0.5)
                            reponse = await self.llm.chat(messages)
                        paragraphs.append(reponse.content)
                    content = ""
                    # Add title to
                    paragraphs.append(item)
            # The last paragraph
            if len(content) > 0:
                content = self.prompt_template_polish.format(content=content)
                messages = [HumanMessage(content)]
                try:
                    reponse = await self.llm.chat(messages)
                except Exception as e:
                    await self._callback_manager.on_llm_error(self, self.llm, e)
                    time.sleep(0.5)
                    reponse = await self.llm.chat(messages)
                paragraphs.append(reponse.content)
            # Generate Citations
            final_report = "\n\n".join(paragraphs)
        else:
            logging.error("Report format error, unable to add abstract and keywords")
            final_report = report
        return final_report

    async def _run(self, report, summarize=None):
        abstract, key = await self.add_abstract(report)
        final_report = await self.polish_paragraph(report, abstract, key)
        await self._callback_manager.on_tool_start(self, tool=self.citation_tool, input_args=final_report)
        if summarize is not None:
            citation_search = add_citation(summarize, self.citation_index_name, self.embeddings)
            final_report, path = await self.citation_tool(
                report=final_report,
                agent_name=self.name,
                report_type=self.report_type,
                dir_path=self.dir_path,
                citation_faiss_research=citation_search,
            )
        path = write_md_to_pdf(self.report_type, self.dir_path, final_report)
        await self._callback_manager.on_tool_end(
            self, tool=self.citation_tool, response={"report": final_report}
        )
        return final_report, path
