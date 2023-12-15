import json
from typing import Optional

from erniebot_agent.agents.base import Agent
from tools.prompt_utils import EB_EDIT_TEMPLATE, eb_functions
from tools.utils import erniebot_chat, json_correct, write_to_json


class EditorActorAgent(Agent):
    DEFAULT_SYSTEM_MESSAGE = EB_EDIT_TEMPLATE

    def __init__(
        self,
        name: str,
        llm: str = "ernie-bot-4",
        system_message: Optional[str] = None,
        config: list = [],
        save_log_path=None,
    ):
        self.name = name
        self.system_message = system_message or self.DEFAULT_SYSTEM_MESSAGE  # type: ignore
        self.model = llm
        self.config = config
        self.save_log_path = save_log_path

    async def _async_run(self, report):
        messages = [
            {
                "role": "user",
                "content": " 草稿为:\n\n" + report,
            }
        ]
        suggestions = erniebot_chat(
            messages=messages, functions=eb_functions, model=self.model, system=self.system_message
        )
        start_idx = suggestions.index("{")
        end_idx = suggestions.rindex("}")
        suggestions = suggestions[start_idx : end_idx + 1]
        suggestions = json_correct(suggestions)
        suggestions = json.loads(suggestions)
        self.config.append(("编辑给出的建议", f"{suggestions}\n\n"))
        self.save_log()
        return suggestions

    def save_log(self):
        write_to_json(self.save_log_path, self.config, mode="a")