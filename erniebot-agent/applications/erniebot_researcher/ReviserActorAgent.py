from typing import Optional

from tools.utils import erniebot_chat, write_to_json

from erniebot_agent.agents.base import Agent
from erniebot_agent.prompt.prompt_template import PromptTemplate


class ReviserActorAgent(Agent):
    DEFAULT_SYSTEM_MESSAGE = """你是一名专业作家。你已经受到编辑的指派，需要修订以下草稿，该草稿由一名非专家撰写。你可以选择是否遵循编辑的备注，视情况而定。
            使用中文输出，只允许对草稿进行局部修改，不允许对草稿进行胡编乱造。
            """

    def __init__(
        self,
        name: str,
        llm: str = "erine-bot-4",
        system_message: Optional[str] = None,
        config: list = [],
        save_log_path=None,
    ):  # type: ignore
        self.name = name
        self.system_message = system_message or self.DEFAULT_SYSTEM_MESSAGE  # type: ignore
        self.model = llm
        self.template = "草稿:\n\n{{draft}}" + "编辑的备注:\n\n{{notes}}"
        self.prompt_template = PromptTemplate(template=self.template, input_variables=["draft", "notes"])
        self.config = config
        self.save_log_path = save_log_path

    async def _async_run(self, draft, notes):
        messages = [
            {
                "role": "user",
                "content": self.prompt_template.format(draft=draft, notes=notes).replace(". ", "."),
            }
        ]
        while True:
            try:
                report = erniebot_chat(messages=messages, system=self.system_message)
                self.config.append(("修订后的报告", report))
                self.save_log()
                return report
            except Exception as e:
                print(e)
                self.config.append(("报错信息", e))
                continue

    def save_log(self):
        write_to_json(self.save_log_path, self.config, mode="a")
