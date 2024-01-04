import json
import logging
import time
from typing import Optional

from tools.utils import ReportCallbackHandler

from erniebot_agent.agents.agent import Agent
from erniebot_agent.agents.schema import AgentResponse
from erniebot_agent.chat_models.erniebot import BaseERNIEBot
from erniebot_agent.memory import HumanMessage
from erniebot_agent.prompt import PromptTemplate

logger = logging.getLogger(__name__)

EB_EDIT_TEMPLATE = """你是一名编辑。
你被指派任务编辑以下草稿，该草稿由一名非专家撰写。
如果这份草稿足够好以供发布，请接受它，或者将它发送进行修订，同时附上指导修订的笔记。
你应该检查以下事项：
- 这份草稿必须充分回答原始问题。
- 这份草稿必须按照APA格式编写。
- 这份草稿必须不包含低级的句法错误。
- 这份草稿的标题不能包含任何引用
如果不符合以上所有标准，你应该发送适当的修订笔记，请以json的格式输出：
如果需要进行修订，则按照下面的格式输出：{"accept":"false","notes": "分条列举出来所给的修订建议。"} 否则输出： {"accept": "true","notes":""}
"""

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


class EditorActorAgent(Agent):
    DEFAULT_SYSTEM_MESSAGE = EB_EDIT_TEMPLATE

    def __init__(
        self,
        name: str,
        llm: BaseERNIEBot,
        llm_long: BaseERNIEBot,
        system_message: Optional[str] = None,
        callbacks=None,
    ):
        self.name = name
        self.system_message = system_message or self.DEFAULT_SYSTEM_MESSAGE
        self.llm = llm
        self.llm_long = llm_long
        self.prompt = PromptTemplate(" 草稿为:\n\n{{report}}", input_variables=["report"])
        if callbacks is None:
            self._callback_manager = ReportCallbackHandler()
        else:
            self._callback_manager = callbacks

    async def run(self, report: str) -> AgentResponse:
        await self._callback_manager.on_run_start(agent=self, prompt=report)
        agent_resp = await self._run(report)
        await self._callback_manager.on_run_end(agent=self, response=agent_resp)
        return agent_resp

    async def _run(self, report):
        await self._callback_manager.on_run_start(self, agent_name=self.name, prompt=report)
        content = self.prompt.format(report=report)
        messages = [HumanMessage(content)]
        retry_count = 0
        while True:
            try:
                if len(content) < 4800:
                    response = await self.llm.chat(
                        messages, functions=eb_functions, system=self.system_message
                    )
                else:
                    response = await self.llm_long.chat(
                        messages, functions=eb_functions, system=self.system_message
                    )
                suggestions = response.content
                start_idx = suggestions.index("{")
                end_idx = suggestions.rindex("}")
                suggestions = suggestions[start_idx : end_idx + 1]
                try:
                    suggestions = json.loads(suggestions)
                except Exception as e:
                    logger.error(e)
                    suggestions = await self.json_correct(suggestions)
                    suggestions = json.loads(suggestions)
                if "accept" not in suggestions and "notes" not in suggestions:
                    raise Exception("accept and notes key do not exist")
                await self._callback_manager.on_run_end(self, agent_name=self.name, prompt=suggestions)
                return suggestions
            except Exception as e:
                logger.error(e)
                await self._callback_manager.on_run_error(self.name, error_information=str(e))
                retry_count += 1
                time.sleep(0.5)
                if retry_count > MAX_RETRY:
                    raise Exception(f"Failed to edit research for {report} after {MAX_RETRY} times.")
                continue

    async def json_correct(self, json_data):
        messages = [HumanMessage("请纠正以下数据的json格式：" + json_data)]
        response = await self.llm.chat(messages)
        res = response.content
        start_idx = res.index("{")
        end_idx = res.rindex("}")
        corrected_data = res[start_idx : end_idx + 1]
        return corrected_data
