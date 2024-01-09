import logging
from typing import Dict, Optional, Union

from tools.utils import JsonUtil, ReportCallbackHandler

from erniebot_agent.agents.callback.callback_manager import CallbackManager
from erniebot_agent.agents.schema import AgentResponse
from erniebot_agent.chat_models.erniebot import BaseERNIEBot
from erniebot_agent.memory import HumanMessage, SystemMessage
from erniebot_agent.prompt.prompt_template import PromptTemplate

logger = logging.getLogger(__name__)
MAX_RETRY = 10
TOKEN_MAX_LENGTH = 4200


class ReviserActorAgent(JsonUtil):
    DEFAULT_SYSTEM_MESSAGE = """你是一名专业作家。你已经受到编辑的指派，需要修订以下草稿，该草稿由一名非专家撰写。你可以选择是否遵循编辑的备注，视情况而定。
    使用中文输出，只允许对草稿进行局部修改，不允许对草稿进行胡编乱造。url链接需要保留，不应该改变"""

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
        self.system_message = (
            system_message.content if system_message is not None else self.DEFAULT_SYSTEM_MESSAGE
        )
        self.llm_long = llm_long
        self.template = "现在给你草稿的内容\n草稿：\n{{draft}}" + "\n请你按照编辑的备注来修改下面的草稿，现在给你编辑的备注：\n{{notes}}"
        self.prompt_template = PromptTemplate(template=self.template, input_variables=["draft", "notes"])
        if callbacks is None:
            self._callback_manager = CallbackManager([ReportCallbackHandler()])
        else:
            self._callback_manager = callbacks

    async def run(self, draft: Union[str, Dict], notes: str) -> AgentResponse:
        if isinstance(draft, dict):
            await self._callback_manager.on_run_start(agent=self, prompt=draft["report"])
        else:
            await self._callback_manager.on_run_start(agent=self, prompt=draft)
        agent_resp = await self._run(draft, notes)
        await self._callback_manager.on_run_end(agent=self, response=agent_resp)
        return agent_resp

    async def _run(self, draft, notes):
        notes = str(notes).replace("{", "").replace("}", "")
        if isinstance(draft, dict):
            content = self.prompt_template.format(draft=draft["report"], notes=notes).replace(". ", ".")
        else:
            content = self.prompt_template.format(draft=draft, notes=notes).replace(". ", ".")
        messages = [HumanMessage(content)]
        retry_count = 0
        while True:
            try:
                if len(content) > TOKEN_MAX_LENGTH:
                    response = await self.llm_long.chat(messages=messages, system=self.system_message)
                else:
                    response = await self.llm.chat(messages=messages, system=self.system_message)
                report = response.content
                if isinstance(draft, dict):
                    draft["report"] = report
                else:
                    draft = report
                return draft
            except Exception as e:
                await self._callback_manager.on_llm_error(self, self.llm, e)
                retry_count += 1
                if retry_count > MAX_RETRY:
                    logger.error("LLM error")
                    return draft
                continue
