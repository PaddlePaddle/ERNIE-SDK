import json
import logging
from typing import Optional

from tools.utils import ReportCallbackHandler

from erniebot_agent.agents.agent import Agent
from erniebot_agent.agents.callback.callback_manager import CallbackManager
from erniebot_agent.agents.schema import AgentResponse
from erniebot_agent.chat_models.erniebot import BaseERNIEBot
from erniebot_agent.memory import HumanMessage, SystemMessage
from erniebot_agent.prompt import PromptTemplate

logger = logging.getLogger(__name__)
TOKEN_MAX_LENGTH = 4200


class RenderAgent(Agent):
    DEFAULT_SYSTEM_MESSAGE = "你是一个报告润色助手，你的主要工作是报告进行内容上的润色"

    def __init__(
        self,
        name: str,
        llm: BaseERNIEBot,
        llm_long: BaseERNIEBot,
        system_message: Optional[SystemMessage] = None,
        callbacks=None,
    ):
        self.name = name
        self.llm = llm
        self.llm_long = llm_long
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

    async def run(self, report: str) -> AgentResponse:
        await self._callback_manager.on_run_start(agent=self, prompt=report)
        agent_resp = await self._run(report)
        await self._callback_manager.on_run_end(agent=self, response=agent_resp)
        return agent_resp

    async def _run(self, report):
        while True:
            try:
                content = self.prompt_template_abstract.format(report=report)
                messages = [HumanMessage(content)]
                if len(content) > TOKEN_MAX_LENGTH:
                    reponse = await self.llm_long.chat(messages)
                else:
                    reponse = await self.llm.chat(messages)
                abstract_json = reponse.content
                l_index = abstract_json.find("{")
                r_index = abstract_json.rfind("}")
                abstract_json = json.loads(abstract_json[l_index : r_index + 1])
                abstract = abstract_json["摘要"]
                key = abstract_json["关键词"]
                if type(key) is list:
                    key = "，".join(key)
                break
            except Exception as e:
                await self._callback_manager.on_llm_error(self, self.llm, e)
                continue
        report_list = report.split("\n\n")
        if "#" in report_list[0] and "##" in report_list[1]:
            paragraphs = []
            title = report_list[0]
            paragraphs.append(title)
            paragraphs.append("**摘要** " + abstract)
            paragraphs.append("**关键词** " + key)
            content = ""
            for item in report_list[1:]:
                if "#" not in item:
                    content += item + "\n"
                else:
                    if len(content) > 300:
                        paragraphs.append(content)
                    elif len(content) > 0:
                        content = self.prompt_template_polish.format(content=content)
                        messages = [HumanMessage(content)]
                        reponse = await self.llm.chat(messages)
                        paragraphs.append(reponse.content)
                    content = ""
                    paragraphs.append(item)
            return "\n\n".join(paragraphs)
        else:
            raise Exception("Report format error")
