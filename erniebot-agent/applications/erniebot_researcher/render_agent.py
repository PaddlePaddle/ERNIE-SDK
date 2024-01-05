import logging
import time
from typing import Optional

from tools.utils import JsonUtil, ReportCallbackHandler, add_citation, write_md_to_pdf

from erniebot_agent.agents.agent import Agent
from erniebot_agent.agents.callback.callback_manager import CallbackManager
from erniebot_agent.agents.schema import AgentResponse
from erniebot_agent.chat_models.erniebot import BaseERNIEBot
from erniebot_agent.memory import HumanMessage
from erniebot_agent.prompt import PromptTemplate

logger = logging.getLogger(__name__)
TOKEN_MAX_LENGTH = 4200


class RenderAgent(Agent, JsonUtil):
    DEFAULT_SYSTEM_MESSAGE = "你是一个报告润色助手，你的主要工作是报告进行内容上的润色"

    def __init__(
        self,
        name: str,
        llm: BaseERNIEBot,
        llm_long: BaseERNIEBot,
        citation_tool,
        embeddings,
        faiss_name_citation: str,
        dir_path: str,
        report_type: str,
        system_message: Optional[str] = None,
        callbacks=None,
    ):
        self.name = name
        self.llm = llm
        self.llm_long = llm_long
        self.report_type = report_type
        self.dir_path = dir_path
        self.embeddings = embeddings
        self.citation = citation_tool
        self.faiss_name_citation = faiss_name_citation
        self.system_message = (
            system_message.content if system_message is not None else self.DEFAULT_SYSTEM_MESSAGE
        )
        self.template_abstract = """
        请你总结报告并给出报告的摘要和关键词，摘要在100-200字之间，关键词不超过5个词。
        你需要输出一个json形式的字符串，内容为{'摘要':...,'关键词':...}。
        现在给你报告的内容：
        {{report}}"""
        self.prompt_template_abstract = PromptTemplate(
            template=self.template_abstract, input_variables=["report"]
        )
        self.template_polish = """你的任务是扩写和润色相关内容，
        你需要把相关内容扩写到300-400字之间，扩写的内容必须与给出的内容相关。
        下面给出内容:
        {{content}}
        扩写并润色内容为:"""
        self.prompt_template_polish = PromptTemplate(
            template=self.template_polish, input_variables=["content"]
        )
        if callbacks is None:
            self._callback_manager = CallbackManager([ReportCallbackHandler()])
        else:
            self._callback_manager = callbacks

    async def run(self, report: str, summarize=None, meta_data=None) -> AgentResponse:
        await self._callback_manager.on_run_start(agent=self, prompt=report)
        agent_resp = await self._run(report, summarize, meta_data)
        await self._callback_manager.on_run_end(agent=self, response=agent_resp)
        return agent_resp

    async def _run(self, report, summarize=None, meta_data=None):
        while True:
            try:
                content = self.prompt_template_abstract.format(report=report)
                messages = [HumanMessage(content)]
                if len(content) > TOKEN_MAX_LENGTH:
                    reponse = await self.llm_long.chat(messages)
                else:
                    reponse = await self.llm.chat(messages)
                res = reponse.content
                abstract_json = self.parse_json(res)
                abstract = abstract_json["摘要"]
                key = abstract_json["关键词"]
                if type(key) is list:
                    key = "，".join(key)
                break
            except Exception as e:
                await self._callback_manager.on_llm_error(self, self.llm, e)
                continue
        report_list = report.split("\n\n")
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
                    # Not to render
                    if len(content) > 300:
                        paragraphs.append(content)
                    # Rendering
                    elif len(content) > 0:
                        content = self.prompt_template_polish.format(content=content)
                        messages = [HumanMessage(content)]
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
        await self._callback_manager.on_tool_start(self, tool=self.citation, input_args=final_report)
        if summarize is not None and meta_data is not None:
            citation_search = add_citation(summarize, self.faiss_name_citation, self.embeddings)
            final_report, path = await self.citation(
                report=final_report,
                meta_data=meta_data,
                agent_name=self.name,
                report_type=self.report_type,
                dir_path=self.dir_path,
                citation_faiss_research=citation_search,
            )
        else:
            path = write_md_to_pdf(self.report_type, self.dir_path, final_report)
        await self._callback_manager.on_tool_end(self, tool=self.citation, response={"report": final_report})
        return final_report, path
