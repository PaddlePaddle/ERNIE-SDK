# Tool 

当你咨询 Agent ：“深圳今天的天气如何？”、“今天下午我有课要上吗？”以及“请帮我请一下明天下午的假，病因是我生病住院”时，Agent 不可能独自完成以上功能，必须要通过外部已有的工具来实现，并且现阶段是可以解决：Tool + Agent。


## Getting Started

每个 Tool 都提供了能够与 Agent 进行交互的能力，用户只需要将 Tool 作为参数传入给 Agent 即可，例如用户想获取当前时间这个功能：

```python
import asyncio
from erniebot_agent.chat_models import ERNIEBot

from erniebot_agent.memory import WholeMemory
from erniebot_agent.tools.tool_manager import ToolManager
from erniebot_agent.agents.functional_agent import FunctionalAgent
from erniebot_agent.tools.current_time_tool.py import CurrentTimeTool

async def main():
    llm = ERNIEBot()
    agent = FunctionalAgent(
        llm=llm,
        tools=[CurrentTimeTool()],
        memory=WholeMemory(),
    )
    result = await agent.async_run("请问现在北京时间是什么时候")
    print(result.text)

asyncio.run(main())
```

## AI STudio ApiHub

AI Studio 的 ApiHub 中提供了种类众多的 Tool，提供给大家调用，访问路径为：`应用` -> `工具` 中即可查看接入的众多 Tool 集合。

比如想要接入百度`文本翻译`的工具，可通过以下步骤实现：

### 获取Access Token

进入：我的 -> 控制台 -> 访问令牌，即可获得对应的 access_token。

### 创建 RemoteToolkit

此时只需要复制其 ID（eg：translation），创建 `RemoteToolkit` 即可使用：

```python
from erniebot_agent.tools import RemoteToolkit
toolkit = RemoteToolkit.from_aistudio("translation", access_token="your-token")
agent = FunctionalAgent(llm=llm, tools=toolkit.get_tools(), memory=WholeMemory())
```

此时即可使用AI Studio ApiHub 中的种类众多的工具集合。