from typing import Any, List, Optional

from erniebot_agent.agents import FunctionalAgent
from erniebot_agent.agents.schema import AgentAction, AgentFile
from erniebot_agent.file_io.base import File
from erniebot_agent.messages import HumanMessage, Message


class PromptAgent(FunctionalAgent):
    def __init__(self, top_k: int = 2, threshold: float = 0.1, token_limit: int = 3000, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k
        self.threshold = threshold
        self.token_limit = token_limit
        # self.system_message = SystemMessage(content="您是一个智能体，旨在回答有关知识库的查询。请始终使用提供的工具回答问题。不要依赖先验知识。")

    async def _async_run(self, prompt: str, files: Optional[List[File]] = None) -> Any:
        actions_taken: List[AgentAction] = []
        files_involved: List[AgentFile] = []
        chat_history: List[Message] = []

        next_step_input = HumanMessage(content=prompt)
        curr_step_output = await self._async_step(
            next_step_input, chat_history, actions_taken, files_involved
        )
        return curr_step_output

    async def _async_step(
        self,
        step_input,
        chat_history: List[Message],
        actions: List[AgentAction],
        files: List[AgentFile],
    ) -> Optional[Any]:
        maybe_action = await self._async_plan(step_input, chat_history)
        if isinstance(maybe_action, AgentAction):
            return True
        else:
            return False
