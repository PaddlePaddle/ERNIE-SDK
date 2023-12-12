import json
from typing import Optional

from erniebot_agent.agents.base import Agent
from tools.prompt_utils import EB_EDIT_TEMPLATE, gpt_functions
from tools.utils import erniebot_chat, json_correct


class EditorActorAgent(Agent):
    DEFAULT_SYSTEM_MESSAGE = EB_EDIT_TEMPLATE

    def __init__(self, name: str, llm: str = "ernie-bot-4", system_message: Optional[str] = None):
        self.name = name
        self.system_message = system_message or self.DEFAULT_SYSTEM_MESSAGE  # type: ignore
        self.model = llm

    async def _async_run(self, report):
        messages = [
            {
                "role": "user",
                "content": " 草稿为:\n\n" + report,
            }
        ]
        suggestions = erniebot_chat(
            messages=messages, functions=gpt_functions, model=self.model, system=self.system_message
        )
        start_idx = suggestions.index("{")
        end_idx = suggestions.rindex("}")
        suggestions = suggestions[start_idx : end_idx + 1]
        suggestions = json_correct(suggestions)
        suggestions = json.loads(suggestions)
        return suggestions
