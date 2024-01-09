import logging
import time
from typing import List, Optional, Union

from tools.utils import JsonUtil, ReportCallbackHandler

from erniebot_agent.chat_models.erniebot import BaseERNIEBot
from erniebot_agent.memory import HumanMessage, Message, SystemMessage
from erniebot_agent.prompt import PromptTemplate

logger = logging.getLogger(__name__)
eb_functions = [
    {
        "name": "revise",
        "description": "发送草稿以进行修订",
        "parameters": {
            "type": "object",
            "properties": {
                "notes": {
                    "type": "string",
                    "description": "编辑的中文备注，用于指导修订。",
                },
            },
        },
    },
    {
        "name": "accept",
        "description": "接受草稿",
        "parameters": {
            "type": "object",
            "properties": {"ready": {"const": True}},
        },
    },
]

MAX_RETRY = 10
TOKEN_MAX_LENGTH = 4200


class EditorActorAgent(JsonUtil):
    DEFAULT_SYSTEM_MESSAGE = """你是一名编辑。
你被指派任务编辑以下草稿，该草稿由一名非专家撰写。
如果这份草稿足够好以供发布，请接受它，或者将它发送进行修订，同时附上指导修订的笔记。
你应该检查以下事项：
- 这份草稿第一行必须是题目，第二行是一级标题。
- 这份草稿的题目前面必须要有#，保证markdown格式正确。
- 这份草稿必须充分回答原始问题。
- 这份草稿必须按照APA格式编写。
- 这份草稿必须不包含低级的句法错误。
- 这份草稿的标题不能包含任何引用
如果不符合以上所有标准，你应该发送适当的修订笔记，请以json的格式输出：
如果需要进行修订，则按照下面的格式输出：{"accept": false,"notes": "分条列举出来所给的修订建议。"} 否则输出： {"accept": true,"notes":""}
"""

    def __init__(
        self,
        name: str,
        llm: BaseERNIEBot,
        llm_long: BaseERNIEBot,
        system_message: Optional[SystemMessage] = None,
        callbacks=None,
    ):
        self.name = name
        self.system_message = (
            system_message.content if system_message is not None else self.DEFAULT_SYSTEM_MESSAGE
        )
        self.llm = llm
        self.llm_long = llm_long
        self.prompt = PromptTemplate(" 草稿为:\n\n{{report}}", input_variables=["report"])
        if callbacks is None:
            self._callback_manager = ReportCallbackHandler()
        else:
            self._callback_manager = callbacks

    async def run(self, report: Union[str, dict[str, str]]) -> dict:
        if isinstance(report, dict):
            report = report["report"]
        await self._callback_manager.on_run_start(agent=self, agent_name=self.name, prompt=report)
        agent_resp = await self._run(report)
        await self._callback_manager.on_run_end(agent=self, response=agent_resp)
        return agent_resp

    async def _run(self, report: Union[dict, str]):
        if isinstance(report, dict):
            report = report["report"]
        content = self.prompt.format(report=report)
        messages: List[Message] = [HumanMessage(content)]
        retry_count = 0
        while True:
            try:
                if len(content) < TOKEN_MAX_LENGTH:
                    response = await self.llm.chat(messages, system=self.system_message)
                else:
                    response = await self.llm_long.chat(messages, system=self.system_message)
                res = response.content

                try:
                    suggestions = self.parse_json(res)
                except Exception as e:
                    logger.error(e)
                    suggestions = await self.json_correct(res)

                if "accept" not in suggestions and "notes" not in suggestions:
                    raise Exception("accept and notes key do not exist")
                return suggestions
            except Exception as e:
                logger.error(e)
                await self._callback_manager.on_llm_error(self, self.llm, error=e)
                retry_count += 1
                time.sleep(0.5)
                if retry_count > MAX_RETRY:
                    raise Exception(f"Failed to edit research for {report} after {MAX_RETRY} times.")
                continue

    async def json_correct(self, json_data):
        messages = [HumanMessage("请纠正以下数据的json格式：%s" % json_data)]
        response = await self.llm.chat(messages)
        res = response.content
        corrected_data = self.parse_json(res)
        return corrected_data
