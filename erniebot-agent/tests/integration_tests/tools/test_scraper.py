from __future__ import annotations

import unittest

from erniebot_agent.agents.function_agent import FunctionAgent
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.memory import WholeMemory
from erniebot_agent.tools.scraper import ScraperTool


class TestScraper(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tool = ScraperTool("eb-agent")

    async def run_query(self, query) -> str:
        llm = ERNIEBot("ernie-3.5")
        agent = FunctionAgent(
            llm=llm,
            tools=[self.tool],
            memory=WholeMemory(),
        )
        result = await agent.run(query)
        return result.text

    async def test_run(self):
        result = await self.run_query(
            "请参考：https://ernie-bot-agent.readthedocs.io/zh-cn/latest/  回答：ERNIE SDK 仓库包含几个项目"
        )
        self.assertIn("ERNIE Bot Agent", result)
