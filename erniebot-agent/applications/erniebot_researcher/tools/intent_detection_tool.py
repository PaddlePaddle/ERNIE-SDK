from __future__ import annotations

from typing import List

from tools.utils import JsonUtil

from erniebot_agent.chat_models.erniebot import BaseERNIEBot
from erniebot_agent.memory import HumanMessage, Message
from erniebot_agent.prompt import PromptTemplate
from erniebot_agent.tools.base import Tool


def auto_agent_instructions():
    agent_instructions = """
        这项任务涉及研究一个给定的主题，不论其复杂性或是否有确定的答案。研究是由一个特定的agent进行的，该agent由其类型和角色来定义，每个agent需要不同的指令。
        Agent: agent是由主题领域和可用于研究所提供的主题的特定agent的名称来确定的。agent根据其专业领域进行分类，每种agent类型都与相应的表情符号相关联。
        示例:
        task: "我应该投资苹果股票吗"
        response:
        {
            "agent": "💰 Finance Agent",
            "agent_role_prompt: "您是一位经验丰富的金融分析AI助手。您的主要目标是根据提供的数据和趋势撰写全面、睿智、公正和系统化的财务报告。"
        }
        task: "转售运动鞋是否有利可图？"
        response:
        {
            "agent":  "📈 Business Analyst Agent",
            "agent_role_prompt": "您是一位经验丰富的AI商业分析助手。您的主要目标是根据提供的商业数据、市场趋势和战略分析制定全面、有见地、公正和系统化的业务报告。"
        }
        task: "海南最有趣的景点是什么？
        response:
        {
            "agent:  "🌍 Travel Agent",
            "agent_role_prompt": "您是一位环游世界的AI导游助手。您的主要任务是撰写有关给定地点的引人入胜、富有洞察力、公正和结构良好的旅行报告，包括历史、景点和文化见解。"
        }
        task: {{content}}
        response:
    """
    return PromptTemplate(agent_instructions, input_variables=["content"])


class IntentDetectionTool(Tool, JsonUtil):
    description: str = "query intent detection tool"

    def __init__(self, llm: BaseERNIEBot) -> None:
        super().__init__()
        self.llm = llm
        self.prompt = auto_agent_instructions()

    async def __call__(self, content: str):
        messages: List[Message] = [HumanMessage(self.prompt.format(content=content))]
        response = await self.llm.chat(messages=messages)
        result = response.content
        result = self.parse_json(result)
        return result
