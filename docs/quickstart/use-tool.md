
`ERNIE Bot Agent` 支持 LocalTool 和 RemoteTool的开发，使用方法如下所示：


## 使用 LocalTool

顾名思义，LocalTool 是一个运行在本地的工具，提供了与 EB 进行交互的能力，接下来展示如何使用  `CurrentTimeTool` 来获取当前的准确时间：

!!! tips

    环境准备请参考：[安装与鉴权](./preparation.md)文档。


```python
import asyncio
from erniebot_agent.agents import FunctionAgent
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.tools.current_time_tool import CurrentTimeTool

async def main():
    agent = FunctionAgent(llm=ERNIEBot(model="ernie-3.5"), tools=[CurrentTimeTool()])
    result = await agent.run("现在北京时间是什么时候？")
    print(result.text)

asyncio.run(main())
```

以上示例展示了如何使用内置的 `CurrentTimeTool` 获取当前时间，开发者也可以自定义 LocalTool，具体请参考 [自定义工具](../modules/tools.md) 文档。


## 使用 RemoteTool

RemoteTool 是运行在远端服务器上的工具，AI Studio 提供了大量的先有 RemoteTool 提供给我们使用，具体可见：[AI Studio 工具中心](https://aistudio.baidu.com/application/center/tool)。

开发者只需要选用对应的工具，复制对应的 tool-id 即可，比如【文本审核】的工具 id 就是 `text-moderation`，然后执行以下代码：

```python
import asyncio
from erniebot_agent.agents import FunctionAgent
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.tools.remote_toolkit import RemoteToolkit

async def main():
    toolkit = RemoteToolkit.from_aistudio("text-moderation")
    agent = FunctionAgent(llm=ERNIEBot(model="ernie-3.5"), tools=toolkit.get_tools())
    result = await agent.run("“欢迎使用ERNIE-Bot-Agent”这句话合规吗？")
    print(result.text)

asyncio.run(main())
```

除此之外如果开发者部署了自定义对应的 RemoteTool，只需要通过 `from_url` 方法接入：

```python
toolkit = RemoteToolkit.from_url("http://127.0.0.1:8000")
```

自定义RemoteTool请参考 [自定义远程工具教程](../cookbooks/agent/remote_tool.ipynb) 文档。
