
# Local Tool

已经介绍过 [tool 快速开始](../modules/tools/getting-started.md) 以及 [如何创建 Tool](../modules/tools/create-tool.md)，本篇文章将通过单词本这个案例详细介绍如何开发智能对话。

## 1. 单词本 Tool

### 1.1 需求描述

创建一个单词本的 LocalTool，可实现添加单词本的功能。

### 1.2 代码实现

```python
from __future__ import annotations

import asyncio
from typing import Any, Dict, Type, List
from pydantic import Field
from erniebot_agent.tools.base import Tool

from erniebot_agent.tools.schema import ToolParameterView

from erniebot_agent.agents.functional_agent import FunctionalAgent
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.memory import WholeMemory


class AddWordInput(ToolParameterView):
    word: str = Field(description="待添加的单词")

class AddWordOutput(ToolParameterView):
    result: str = Field(description="表示是否成功将单词成功添加到词库当中")

class AddWordTool(Tool):
    description: str = "添加单词到词库当中"
    input_type: Type[ToolParameterView] = AddWordInput
    ouptut_type: Type[ToolParameterView] = AddWordOutput

    def __init__(self) -> None:
        self.word_books = {}
        super().__init__()

    async def __call__(self, word: str) -> Dict[str, Any]:
        if word in self.word_books:
            return {"result": f"<{word}>单词已经存在，无需添加"}
        self.word_books[word] = True
        words = "\n".join(list(self.word_books.keys()))
        return {"result": f"<{word}>单词已添加成功, 当前单词本中有如下单词：{words}"}

async def main():
    agent = FunctionalAgent(ERNIEBot("ernie-3.5"), tools=[AddWordTool()], memory=WholeMemory())
    result = await agent.async_run("将单词：“red”添加到单词本当中")
    print(result)

asyncio.run(main())

```

### 1.3 输出示例

```shell
AgentResponse(text='恭喜您，单词“red”已成功添加到单词本中。目前单词本中有如下单词：red。如果您需要继续添加其他单词，请随时告诉我。', chat_history=[<HumanMessage role: 'user', content: '将单词：“red”添加到单词当中', token_count: 112>, <AIMessage role: 'assistant', function_call: {'name': 'AddWordTool', 'thoughts': '用户想要将单词“red”添加到词库当中；我需要使用AddWordTool工具来实现这一需求；根据AddWordTool工具的定义，全部参数集合为[word]；其中"required": true的必要参数集合为[word]；结合用户当前问题“将单词：“red”添加到单词当中”和整个对话历史，用户已经提供了以下参数值{word: \'red\'}；其中已经提供对应参数值的"required": true的必要参数集合为[word]；尚未提供对应参数值的"required": true参数列表为[]；由于尚未提供对应参数值的"required": true参数列表为[]，即全部"required": true的必要参数都已经提供，我可以直接调用工具AddWordTool', 'arguments': '{"word":"red"}'}, token_count: 157>, <FunctionMessage role: 'function', name: 'AddWordTool', content: '{"result": "<red>单词已添加成功, 当前单词本中有如下单词：red"}'>, <AIMessage role: 'assistant', content: '恭喜您，单词“red”已成功添加到单词本中。目前单词本中有如下单词：red。如果您需要继续添加其他单词，请随时告诉我。', token_count: 35>], actions=[AgentAction(tool_name='AddWordTool', tool_args='{"word":"red"}')], files=[], status='FINISHED')
```

通过以上代码即可实现添加单词的LocalTool。
