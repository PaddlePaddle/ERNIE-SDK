from __future__ import annotations

import asyncio
import os

import json
from typing import Any, Dict, Type, List
from pydantic import Field
from datetime import datetime
import pytest
import unittest
from erniebot_agent.tools.base import Tool

from erniebot_agent.tools.remote_toolkit import RemoteToolkit
from erniebot_agent.tools.schema import ToolParameterView

from erniebot_agent.agents.functional_agent import FunctionalAgent
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.tools.tool_manager import ToolManager
from erniebot_agent.memory import WholeMemory


from erniebot_agent.tools.chat_with_eb import ChatWithEB

WORD_BOOK_FILE = "./words.json"

def read_word_book():
    if not os.path.exists(WORD_BOOK_FILE):
        write_word_book({})
    with open(WORD_BOOK_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def write_word_book(word_book: dict):
    with open(WORD_BOOK_FILE, "w", encoding="utf-8") as f:
        json.dump(word_book, f, ensure_ascii=False, indent=8)


class AddWordInput(ToolParameterView):
    word: str = Field(description="待添加的单词")
    score: int = Field(description="单词分数：非常熟悉分数为 0；熟悉分数为1；不熟悉分数为 2，默认值为 0", default=0)

class AddWordOutput(ToolParameterView):
    result: str = Field(description="表示是否成功将单词成功添加到词库当中")

class AddWordTool(Tool):
    description: str = "添加单词到词库当中"
    input_type: Type[ToolParameterView] = AddWordInput
    ouptut_type: Type[ToolParameterView] = AddWordOutput


    async def __call__(self, word: str, score: int) -> Dict[str, Any]:
        word_book = read_word_book()
        if word in word_book:
            return {"result": f"<{word}>单词已经存在，无需添加"}
        word_book[word] = {
            "create_time": datetime.now().strftime("%Y年%m月%d日 %H时%M分%S秒"),
            "score": score
        }
        write_word_book(word_book)
        return {"result": f"<{word}>单词已添加成功"}

class Word(ToolParameterView):
    word: str = Field(description="单词")
    create_time: str = Field(description="创建时间")
    score: int = Field(description="单词分数：非常熟悉分数为 0；熟悉分数为1；不熟悉分数为 2")

class GetWordBookInput(ToolParameterView):
    score: int = Field(description="单词记忆的熟悉程度", default=0)
    

class GetWordBookOutput(ToolParameterView):
    words: List[Word] = Field(description="词库中的单词列表")


class GetWordTool(Tool):
    description: str = "获取词库中的单词列表"
    
    async def __call__(self, score: int = 0) -> Dict[str, Any]:
        word_book = read_word_book()
        word_book = {key: value for key, value in word_book.items() if value["score"] >= score}

        words = [{"word": key, **value} for key, value in word_book.items()]
        return {"words": words}
        

class DeleteWordInput(ToolParameterView):
    word: str = Field(description="待删除的单词")

class DeleteWordTool(Tool):
    description: str = "删除词库中的单词"
    input_type: Type[ToolParameterView] = DeleteWordInput

    async def __call__(self, word: str) -> Dict[str, Any]:
        word_book = read_word_book()
        if word not in word_book:
            return {"result": f"单词：<{word}>不存在，无需删除"}
        
        word_book.pop(word)
        write_word_book(word_book)
        return {"result": f"已将单词：<{word}>从词库中删除"}
    

async def main():
    agent = FunctionalAgent(
        ERNIEBot("ernie-3.5"), 
        tools=[AddWordTool()], memory=WholeMemory()
    )
    while True:
        prompt = input("Human:")
        result = await agent.async_run(prompt)
        print("Bot: ", result.text)


asyncio.run(main())