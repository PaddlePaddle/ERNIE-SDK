import logging
from typing import Optional

from tools.utils import ReportCallbackHandler

from erniebot_agent.agents.agent import Agent
from erniebot_agent.agents.schema import AgentResponse
from erniebot_agent.chat_models.erniebot import BaseERNIEBot
from erniebot_agent.memory import HumanMessage
from erniebot_agent.prompt.prompt_template import PromptTemplate

logger = logging.getLogger(__name__)


class ReviserActorAgent(Agent):
    DEFAULT_SYSTEM_MESSAGE = """你是一名专业作家。你已经受到编辑的指派，需要修订以下草稿，该草稿由一名非专家撰写。你可以选择是否遵循编辑的备注，视情况而定。
            使用中文输出，只允许对草稿进行局部修改，不允许对草稿进行胡编乱造。
            """

    def __init__(
        self,
        name: str,
        llm: BaseERNIEBot,
        system_message: Optional[str] = None,
        callbacks=None,
    ):
        self.name = name
        self.llm = llm
        self.system_message = system_message or self.DEFAULT_SYSTEM_MESSAGE
        self.model = llm
        self.template = "草稿:\n\n{{draft}}" + "编辑的备注:\n\n{{notes}}"
        self.prompt_template = PromptTemplate(template=self.template, input_variables=["draft", "notes"])
        if callbacks is None:
            self._callback_manager = ReportCallbackHandler()
        else:
            self._callback_manager = callbacks

    async def run(self, draft: str, notes: str) -> AgentResponse:
        await self._callback_manager.on_run_start(agent=self, prompt=draft)
        agent_resp = await self._run(draft, notes)
        await self._callback_manager.on_run_end(agent=self, response=agent_resp)
        return agent_resp

    async def _run(self, draft, notes):
        await self._callback_manager.on_run_start(self, agent_name=self.name, prompt=draft)
        messages = [HumanMessage(self.prompt_template.format(draft=draft, notes=notes).replace(". ", "."))]
        while True:
            try:
                response = await self.llm.chat(messages=messages, system=self.system_message)
                report = response.content
                await self._callback_manager.on_run_end(self, agent_name=self.name, prompt=report)
                return report
            except Exception as e:
                logger.error(e)
                await self._callback_manager.on_run_error(self.name, str(e))
                continue
