# ChatModels模块

## 1. 简介

文心一言是百度研发的知识增强大语言模型，能够与人对话互动，回答问题，协助创作，高效便捷地帮助人们获取信息、知识和灵感。

大家通过`ERNIE Bot SDK`可以调用文心一言模型完成简单的AI任务，但是如果面对复杂的场景应用，可能需要基于`ERNIE Bot SDK`开发较多的功能模块。

为了降低使用门槛和开发工作，我们推荐通过`ERNIE Bot Agent`来调用文心一言模型，助力大家快速开发出AI应用。

`ERNIE Bot Agent`支持多个文心一言模型，包括`ernie-3.5`、`ernie-turbo`、`ernie-4.0`和`ernie-longtext`。

| 模型名称 | 说明 | 功能 | 输入token数量上限 |
|:--- | :--- | :--- | :--- |
| ernie-3.5 | 文心大模型3.5版本。具备优秀的知识增强和内容生成能力，在文本创作、问答、推理和代码生成等方面表现出色。 | 对话补全，函数调用 | 3000 |
| ernie-turbo | 文心大模型。相比ernie-3.5模型具备更快的响应速度和学习能力，API调用成本更低。 | 对话补全 |  3000 |
| ernie-4.0 | 文心大模型4.0版本，具备目前系列模型中最优的理解和生成能力。 | 对话补全，函数调用 |  3000 |
| ernie-longtext | 文心大模型。在ernie-3.5模型的基础上增强了对长对话上下文的支持，输入token数量上限为7000。 | 对话补全，函数调用 |  7000 |

## 2. 核心类

下面简单介绍`ChatModels`模块的核心类，详细接口请参考[API文档](../../package/erniebot_agent/chat_models/)。

`ChatModel`类是基类，以下是主要的属性和方法。

| 属性       | 类型           | 描述                                                      |
| ---------- | -------------- | ------------------------------------------------------- |
| model         | str          | 文心一言模型的名称，支持"ernie-3.5", "ernie-turbo", "ernie-4.0", "ernie-longtext"   |
| default_chat_kwargs | Dict[str, Any] | 设置文心一言模型的默认参数，比如`_config_`, `top_p`等等                       |

| 方法              | 描述                                                                  |
| ----------------- | -------------------------------------------------------------------- |
| chat           | 和模型进行多轮对话，是一个虚方法                                              |


`ERNIEBot`类继承`ChatModel`类，以下是主要的属性和方法。

| 属性       | 类型           | 描述                                                      |
| ---------- | -------------| --------------------------------------------------------- |
| model      | str          | 文心一言模型的名称，支持"ernie-3.5", "ernie-turbo", "ernie-4.0", "ernie-longtext"  |
| api_type   | str          | 文心一言模型的后端，支持"aistudio"和"qianfan"，默认是"aistudio"。                    |
| access_token | Optional[str]  | 文心一言模型的鉴权access token，不同后端需要使用对应的access token                |
| enable_multi_step_tool_call | bool  | 设置开启多轮调用tool的功能                                               |
| default_chat_kwargs | Dict[str, Any] | 设置文心一言模型的默认参数，比如`_config_`, `top_p`等等                   |

| 方法               | 描述                                                                  |
| ----------------- | --------------------------------------------------------------------  |
| chat              | 和模型进行多轮对话                                                       |

!!! notes 注意

    不同后端需要使用对应的access token进行鉴权，我们推荐使用"aistudio"后端，文档的示例也都使用"aistudio"后端进行演示。

## 3. 使用示例

为了直观展示，我们举例进行说明，请先确保完成`ERNIE Bot Agent`的安装和鉴权步骤。

出于使用场景和性能的考虑，`ERNIE Bot Agent`只提供异步接口来使用文心一言模型。

### 3.1 进行文本补全

这个示例中，首先创建文心一言`ernie-3.5`模型，然后两次调用`chat`接口传入只有单条`HumanMessage`的数组，文心一言模型会对单条`HumanMessage`做出回答，返回一条`AIMessage`。

```python
import os
import asyncio
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.memory import HumanMessage

os.environ["EB_AGENT_ACCESS_TOKEN"] = "your access token"

async def demo():
    model = ERNIEBot(model="ernie-3.5")
    human_message = HumanMessage(content='你好，你是谁')
    ai_message = await model.chat(messages=[human_message])
    print(ai_message.content, '\n')

    human_message = HumanMessage(content='推荐三个深圳有名的景点')
    ai_message = await model.chat(messages=[human_message],
                                        stream=True)  # 流式返回
    async for chunk in ai_message:
        print(chunk.content, end='')

asyncio.run(demo())
```

示例的输出类似于：
```
你好，我是百度公司开发的人工智能语言模型，我的中文名是文心一言，英文名是ERNIE Bot。如果您有任何问题，请随时向我提问。

深圳有许多著名的景点，以下是三个推荐景点：
1. 深圳世界之窗
2. 深圳欢乐谷
3. 深圳东部华侨城
```

### 3.2 进行多轮对话

如果希望进行多轮对话，而且让文心一言模型能够根据上下文进行回答，可以执行如下代码。其中前一轮对话的输入输出`Message`会被带入第二轮对话。

```python
import os
import asyncio
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.memory import HumanMessage

os.environ["EB_AGENT_ACCESS_TOKEN"] = "your access token"

async def demo():
    model = ERNIEBot(model="ernie-3.5")
    messages = []

    messages.append(HumanMessage(content='推荐三个深圳有名的景点'))
    ai_message = await model.chat(messages=messages)
    messages.append(ai_message)
    print(ai_message.content, '\n')

    messages.append(HumanMessage(content='根据你推荐的景点，帮我做一份一日游的攻略'))
    ai_message = await model.chat(messages=messages)
    messages.append(ai_message)
    print(ai_message.content, '\n')

asyncio.run(demo())
```

示例的输出类似于：
```
深圳有很多有名的景点，以下是三个推荐的景点：
1. **深圳世界之窗**：
2. **深圳欢乐谷**：
3. **深圳东部华侨城**：

好的，以下是一份深圳一日游的攻略：
早上：
* 早上9点左右到达深圳世界之窗。首先可以参观非洲区的莫高窟、埃塞俄比亚院及四大文明古国馆，了解不同文化的历史和特点。
* 然后可以前往亚洲区的比萨斜塔、悉尼歌剧院等著名建筑，感受不同国家的建筑风格和文化内涵。
* 接着可以参观欧洲区的罗马斗兽场、白宫等著名景点，了解不同国家的政治、历史和文化。

中午：
* 在世界之窗内的餐厅享用午餐，品尝当地美食。
下午：
* 下午可以前往深圳欢乐谷，游览各种刺激和好玩的游乐设施。可以先体验一下高速过山车、云霄飞车等刺激的项目，然后再尝试其他的游乐设施。
* 可以选择在欢乐谷内游玩一整个下午，尽情享受游乐园的乐趣。
晚上：
* 晚上可以选择在东部华侨城内度过。可以先去温泉浴场放松一下身心，然后再去主题公园欣赏各种表演和娱乐活动。
```

### 3.3 命令行聊天应用

下面示例实现了一个简易的命令行聊天应用。

```python
import os
import asyncio
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.memory import HumanMessage, AIMessage

os.environ["EB_AGENT_ACCESS_TOKEN"] = "your access token"

async def demo():
    model = ERNIEBot(model='ernie-3.5')
    messages = []

    print('你好，有什么我可以帮助你的吗？')
    while True:
        prompt = input()
        messages.append(HumanMessage(prompt))
        ai_message = await model.chat(messages=messages, stream=True)

        result = ''
        async for chunk in ai_message:
            result += chunk.content
            print(chunk.content, end='')
        print('')
        messages.append(AIMessage(result))

asyncio.run(demo())
```