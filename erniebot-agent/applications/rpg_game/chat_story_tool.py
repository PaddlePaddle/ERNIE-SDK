from typing import Type

from pydantic import Field

from erniebot_agent.agents import Agent
from erniebot_agent.tools.base import Tool, ToolParameterView


class ChatStoryToolInputView(ToolParameterView):
    query: str = Field(description="用户的指令")


class ChatStoryToolOutputView(ToolParameterView):
    return_story: str = Field(description="生成的包括<场景描述>、<场景图片>和<选择>的互动内容")


class ChatStoryTool(Tool):
    description: str = "结合用户的选择、{GAME}背景故事以及玩家角色，按要求生成接下来的故事情节"
    input_type: Type[ToolParameterView] = ChatStoryToolInputView
    ouptut_type: Type[ToolParameterView] = ChatStoryToolOutputView

    def __init__(self, agent, game: str) -> None:
        super().__init__()
        self.agent: Agent = agent
        self.description = self.description.format(GAME=game)

    async def __call__(self, query: str) -> str:
        response = await self.agent.run(query)
        return {"return_story": response.text}
