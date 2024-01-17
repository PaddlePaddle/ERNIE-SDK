import json
from typing import List, Optional, Sequence

from erniebot_agent.agents.agent import Agent
from erniebot_agent.agents.schema import AgentResponse, AgentStep, File
from erniebot_agent.memory.messages import HumanMessage, Message
from erniebot_agent.tools.base import BaseTool


class PromptAgent(Agent):
    async def _run(self, prompt: str, files: Optional[Sequence[File]] = None) -> AgentResponse:
        chat_history: List[Message] = []
        steps_taken: List[AgentStep] = []
        run_input = await HumanMessage.create_with_files(
            prompt, files or [], include_file_urls=self.file_needs_url
        )
        chat_history.append(run_input)
        msg = await self._step(chat_history)
        text = json.dumps({"msg": msg}, ensure_ascii=False)
        response = self._create_stopped_response(chat_history, steps_taken, message=text)
        return response

    async def _step(self, chat_history: List[Message], selected_tool: Optional[BaseTool] = None) -> bool:
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

    def _create_stopped_response(
        self,
        chat_history: List[Message],
        steps: List[AgentStep],
        message: str,
    ) -> AgentResponse:
        return AgentResponse(
            text=message,
            chat_history=chat_history,
            steps=steps,
            status="STOPPED",
        )
