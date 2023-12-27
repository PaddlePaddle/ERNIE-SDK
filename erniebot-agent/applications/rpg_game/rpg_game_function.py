import os
from typing import Optional, List, Type, Dict
from pydantic import BaseModel, Field
from erniebot_agent.extensions.langchain.embeddings import ErnieEmbeddings
from erniebot_agent.tools.base import Tool, ToolParameterView
from erniebot_agent.agents import FunctionAgent
from erniebot_agent.memory import WholeMemory, SlidingWindowMemory
from erniebot_agent.chat_models.erniebot import ERNIEBot
from erniebot_agent.memory.messages import AIMessage, HumanMessage, Message, SystemMessage, FunctionMessage

INSTRUCTION = """你的指令是为我提供一个基于《{SCRIPT}》剧情的在线RPG游戏体验。在这个游戏中，玩家将扮演《{SCRIPT}》剧情关键角色，游戏情景将基于《{SCRIPT}》剧情。\
这个游戏的玩法是互动式的，并遵循以下特定格式：

<场景描述>：根据玩家的选择，故事情节将按照《{SCRIPT}》剧情的线索发展。你将描述角色所处的环境和情况，不得少于三句话。

<场景图片>：对于每个场景，你将创造一个概括该情况的图像。用于提供给调用画图工具(ImageGenerateTool)。

<选择>：在每次互动中，你将为玩家提供三个行动选项，分别标为1、2、3，以及第四个选项“输入玩家自定义的选择”。故事情节将根据玩家选择的行动进展。\
如果一个选择不是直接来自《{SCRIPT}》剧情，你将创造性地适应故事，最终引导它回归原始情节。

整个故事将围绕《{SCRIPT}》丰富而复杂的世界展开。每次互动必须包括<场景描述>、<场景图片>和<选择>。所有内容将以中文呈现。\
你的重点将仅仅放在提供场景描述，场景图片和选择上，不包含其他游戏指导。场景尽量不要重复，要丰富一些。

当我说游戏开始的时候，开始游戏。每次只要输出一组互动，不要自己生成互动。"""

class ChatStoryToolInputView(ToolParameterView):
    query: str = Field(description="用户的指令")

class ChatStoryToolOutputView(ToolParameterView):
    return_story: str = Field(description="生成的包括<场景描述>、<场景图片>和<选择>的互动内容")

class ChatStoryTool(Tool):
    description: str = "结合用户的选择、背景故事以及玩家角色，按要求生成接下来的故事情节"
    input_type: Type[ToolParameterView] = ChatStoryToolInputView
    ouptut_type: Type[ToolParameterView] = ChatStoryToolOutputView

    def __init__(self, agent) -> None:
        super().__init__()
        self.agent = agent
    
    async def __call__(self, query: str) -> str:
        response = await self.agent.run(query)
        return {"return_story": response.text}

def creates_story_tool():
    memory = SlidingWindowMemory(max_round=2)
    llm = ERNIEBot(model="ernie-3.5", api_type='aistudio')
    agent = FunctionAgent(llm=llm, tools=[], system_message=SystemMessage(INSTRUCTION.format(SCRIPT="仙剑奇侠传"), memory=memory))
    tool = ChatStoryTool(agent)
    return tool


story_tool = creates_story_tool()
SYSTEM_MESSAGE =  "你是《{SCRIPT}》沉浸式图文RPG场景助手，能够生成图文剧情。\
                请你先调用ChatStoryTool生成互动，然后调用ImageGenerateTool生成图片，\
                最后将图片描述的部分替换为fileid，其他的内容不要改变。"
# 创建一个ERNIEBot实例，使用"ernie-bot-8k"模型。
llm = ERNIEBot(model="ernie-3.5", api_type='aistudio', enable_multi_step_tool_call=True)
memory = WholeMemory()
agent = FunctionAgent(llm=llm, tools=[story_tool, img_tool], memory=memory, system_message=SystemMessage(SYSTEM_MESSAGE.format(SCRIPT="仙剑奇侠传")))
os.environ['EB_AGENT_LOGGING_LEVEL'] = 'info'
query = '开始游戏'
response = await agent_all.async_run(query)