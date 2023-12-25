# Tool 

当你咨询 Agent ：“深圳今天的天气如何？”、“今天下午我有课要上吗？”以及“请帮我请一下明天下午的假，病因是我生病住院”时，Agent 不可能独自完成以上功能，必须要通过与外部已有的工具交互来实现，当前解决方案是：Tool + Agent。

## Getting Started

每个 `Tool` 都提供了能够与 `Agent` 进行交互的能力，用户只需将 `Tool` 作为参数传入给 `Agent`，在对话的过程中将会根据用户的`query`判别是否要调用对应的 `Tool`，并将工具执行的结果润色成自然语言，例如用户想获取当前时间这个功能：

```python
# 1. import 相关的类
import asyncio
from erniebot_agent.chat_models import ERNIEBot

from erniebot_agent.memory import WholeMemory
from erniebot_agent.tools.tool_manager import ToolManager
from erniebot_agent.agents.functional_agent import FunctionalAgent
from erniebot_agent.tools.current_time_tool.py import CurrentTimeTool

async def main():
    # 2. 定义 Agent
    llm = ERNIEBot()
    agent = FunctionalAgent(
        llm=llm,
        tools=[CurrentTimeTool()],
        memory=WholeMemory(),
    )

    # 3. 执行用户 query
    result = await agent.async_run("请问现在北京时间是什么时候")
    print(result.text)
    # 现在的时间是2023 年 10 月 25 日晚上8 点 15 分。

asyncio.run(main())
```

## AI Studio 工具中心

`AI Studio` 的工具中心提供了种类众多的 `Tool`，提供给大家调用，访问路径为：`应用` -> `工具` 中即可查看接入的众多 `Tool` 集合。

比如接入百度`文本翻译`的工具，可通过以下步骤实现：

### 1. 获取Access Token

进入：我的 -> 控制台 -> 访问令牌，即可获得对应的 access_token。

### 2. 创建 RemoteToolkit

此时只需要复制其 ID（eg：translation），创建 `RemoteToolkit` 即可使用：

```python
from erniebot_agent.tools import RemoteToolkit
toolkit = RemoteToolkit.from_aistudio("translation", access_token="your-token")
agent = FunctionalAgent(llm=llm, tools=toolkit.get_tools(), memory=WholeMemory())
```

此时即可使用AI Studio 工具中心的 `translation` 工具。