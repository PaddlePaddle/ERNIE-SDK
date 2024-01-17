from typing import Any, List, Optional, Sequence, Tuple

from erniebot_agent.agents import FunctionalAgent
from erniebot_agent.agents.schema import AgentAction, AgentFile, File, AgentResponse, AgentStep
from erniebot_agent.memory.messages import HumanMessage, Message
from erniebot_agent.tools.base import BaseTool

class PromptAgent(FunctionalAgent):

    async def _run(self, prompt: str, files: Optional[Sequence[File]] = None) -> AgentResponse:
        actions_taken: List[AgentAction] = []

        chat_history: List[Message] = []

        next_step_input = HumanMessage(content=prompt)
        curr_step_output = await self._step(
            next_step_input, chat_history, actions_taken
        )
        return curr_step_output

    async def _step(
        self, chat_history: List[Message], selected_tool: Optional[BaseTool] = None
    ) -> Tuple[AgentStep, List[Message]]:
        new_messages: List[Message] = []
        input_messages = self.memory.get_messages() + chat_history
        if selected_tool is not None:
            tool_choice = {"type": "function", "function": {"name": selected_tool.tool_name}}
            llm_resp = await self.run_llm(
                messages=input_messages,
                functions=[selected_tool.function_call_schema()],  # only regist one tool
                tool_choice=tool_choice,
            )
        else:
            llm_resp = await self.run_llm(messages=input_messages)

        output_message = llm_resp.message  # AIMessage
        new_messages.append(output_message)
        if output_message.function_call is not None:
            return True
        else:
            return False
