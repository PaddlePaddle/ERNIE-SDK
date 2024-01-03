import json
import logging
from typing import Optional

from tools.prompt_utils import eb_functions
from tools.utils import erniebot_chat, json_correct, write_to_json

from erniebot_agent.agents.agent import Agent
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


class EditorActorAgent(Agent):
    DEFAULT_SYSTEM_MESSAGE = EB_EDIT_TEMPLATE

    def __init__(
        self,
        name: str,
        llm: str = "ernie-4.0",
        system_message: Optional[str] = None,
        config: list = [],
        save_log_path=None,
    ):
        self.name = name
        self.system_message = system_message or self.DEFAULT_SYSTEM_MESSAGE  # type: ignore
        self.model = llm
        self.config = config
        self.save_log_path = save_log_path
        self.prompt = PromptTemplate(" 草稿为:\n\n{{report}}", input_variables=["report"])

    async def _run(self, report):
        messages = [
            {
                "role": "user",
                "content": self.prompt.format(report=report),
            }
        ]
        while True:
            try:
                suggestions = erniebot_chat(
                    messages=messages, functions=eb_functions, model=self.model, system=self.system_message
                )
                start_idx = suggestions.index("{")
                end_idx = suggestions.rindex("}")
                suggestions = suggestions[start_idx : end_idx + 1]
                suggestions = json_correct(suggestions)
                suggestions = json.loads(suggestions)
                if "accept" not in suggestions and "notes" not in suggestions:
                    raise Exception("accept and notes key do not exist")

                self.config.append(("编辑给出的建议", f"{suggestions}\n\n"))
                self.save_log()
                return suggestions
            except Exception as e:
                logger.error(e)
                self.config.append(("报错信息", e))
                continue

    def save_log(self):
        write_to_json(self.save_log_path, self.config, mode="a")
