from __future__ import annotations

import unittest

from erniebot_agent.agents.function_agent import FunctionAgent
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.memory import WholeMemory
from erniebot_agent.tools.scraper import ScraperTool


class TestScraper(unittest.IsolatedAsyncioTestCase):

    async def test_run(self):
        tool = ScraperTool("eb-agent")
        
        content = await tool(
            urls=[
                {"url": "https://ernie-bot-agent.readthedocs.io/zh-cn/latest/"},
                {"url": "https://arxiv.org/pdf/1810.04805.pdf"},
                {"url": "https://arxiv.org/abs/2005.14165"},
            ]
        )

        result_0 = content["result"][0]["raw_content"]
        self.assertIn("ERNIE Bot Agent", result_0)
        self.assertIn("ERNIE Bot", result_0)

        result_1 = content["result"][1]["raw_content"]
        self.assertIn("BERT: Pre-training of Deep Bidirectional Transformers", result_1)

        result_2 = content["result"][2]["raw_content"]
        self.assertIn("GPT-3 in general.", result_2)
