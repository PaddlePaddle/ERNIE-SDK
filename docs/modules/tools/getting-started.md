# Tool 

当您向Agent询问“深圳今天的天气如何？”、“今天下午我有课要上吗？”或者“请帮我请一下明天下午的假，病因是我生病住院”时，Agent无法仅凭自身能力提供完整的回答或执行相应的操作。为了完成这些任务，Agent需要与外部工具（Tool）进行交互。目前，实现这些功能的解决方案是采用“Tool+Agent”的组合。通过这种组合，Agent可以调用外部工具来获取天气信息、检查您的课程安排或提交请假申请。这种协同工作的方式使得Agent能够更全面地满足您的需求。

## Getting Started

每个“工具”（Tool）都提供了与“代理”（Agent）进行交互的能力。用户只需将所需的“工具”作为参数传递给“代理”。在对话过程中，系统将根据用户的查询内容（query）判断是否需要调用相应的“工具”，并将该工具执行的结果转化为自然语言进行输出。例如，用户若想获取当前时间，系统能够调用相应的工具来提供这一功能。

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

AI Studio的工具中心提供了丰富多样的Tool供用户调用。要查看这些已接入的Tool集合，只需按照以下路径进行访问：应用 -> 工具。

以接入百度文本翻译工具为例，以下是实现步骤：

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