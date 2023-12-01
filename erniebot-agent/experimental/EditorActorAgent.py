from typing import Optional

from erniebot_agent.agents.base import Agent
from erniebot_agent.chat_models.erniebot import ERNIEBot
from erniebot_agent.prompt.prompt_template import PromptTemplate


class EditorActorAgent(Agent):
    DEFAULT_SYSTEM_MESSAGE = """你是一名编辑。
        你被指派任务编辑以下草稿，该草稿由一名非专家撰写。
        如果这份草稿足够好以供发布，请接受它，或者将它发送进行修订，同时附上指导修订的笔记。
        你应该检查以下事项：
        - 这份草稿必须充分回答原始问题。
        - 这份草稿必须按照APA格式编写。
        - 这份草稿必须不包含低级的句法错误。
        如果不符合以上所有标准，你应该发送适当的修订笔记，请以json的格式输出：
        如果需要进行修订，则按照下面的格式输出：{"accept":"false","notes": "分条列举出来所给的修订建议。"} 否则输出： {"accept": "true","notes":""}
        """

    def __init__(self, name: str, llm: ERNIEBot, system_message: Optional[str] = None):
        self.name = name
        self.system_message = system_message or self.DEFAULT_SYSTEM_MESSAGE  # type: ignore
        self.model = llm
        self.prompt = PromptTemplate("草稿:\n\n{draft}", input_variables=["draft"])

        # self.functions = [
        #     {
        #         "name": "revise",
        #         "description": "Sends the draft for revision",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "notes": {
        #                     "type": "string",
        #                     "description": "The editor's notes to guide the revision.",
        #                 },
        #             },
        #         },
        #     },
        #     {
        #         "name": "accept",
        #         "description": "Accepts the draft",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {"ready": {"const": True}},
        #         },
        #     },
        # ]

    def run(self, report):
        # return self.model(self.prompt_template.format(report), functions = self.functions)
        return self.model(self.prompt_template.format(draft=report), system=self.system_message)
