from typing import Optional

from erniebot_agent.agents.base import Agent
from erniebot_agent.chat_models.erniebot import ERNIEBot
from erniebot_agent.prompt.prompt_template import PromptTemplate


class ReviserActorAgent(Agent):
    DEFAULT_SYSTEM_MESSAGE = """你是一名专业作家。你已经受到编辑的指派，需要修订以下草稿，该草稿由一名非专家撰写。你可以选择是否遵循编辑的备注，视情况而定。
            使用中文输出，只允许对草稿进行局部修改，不允许对草稿进行胡编乱造。
            """

    def __init__(self, name: str, llm: ERNIEBot, system_message: Optional[str] = None):  # type: ignore
        self.name = name
        self.system_message = system_message or self.DEFAULT_SYSTEM_MESSAGE  # type: ignore
        self.model = llm
        self.template = "草稿:\n\n{draft}" + "编辑的备注:\n\n{notes}"
        self.prompt_template = PromptTemplate(template=self.template, input_variables=["draft", "notes"])

    async def _async_run(self, draft, notes):
        return self.model(self.prompt_template.format(draft, notes), system=self.system_message)
