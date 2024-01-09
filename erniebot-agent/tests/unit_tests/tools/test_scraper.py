from __future__ import annotations

import unittest

from typing import List
from erniebot_agent.tools.scraper import ScraperTool


class TestScraper(unittest.IsolatedAsyncioTestCase):

    def setUp(self) -> None:
        self.tool = ScraperTool("eb-agent")

    async def _run_tool(self, url: str, keywords: List[str]):
        content = await self.tool(
            urls=[{"url": url}]
        )
        raw_content = content["result"][0]["raw_content"]
        for keyword in keywords:
            self.assertIn(keyword, raw_content)
        
    async def test_run(self):
        await self._run_tool(
            "https://ernie-bot-agent.readthedocs.io/zh-cn/latest/",
            ["ERNIE Bot Agent", "ERNIE Bot"]
        )

        await self._run_tool(
            "https://arxiv.org/pdf/1810.04805.pdf",
            ["BERT: Pre-training of Deep Bidirectional Transformers"]
        )
        await self._run_tool(
            "https://arxiv.org/abs/2005.14165",
            ["GPT-3 in general."]
        )