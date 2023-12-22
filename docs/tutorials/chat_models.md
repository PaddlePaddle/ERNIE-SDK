# ChatModels模块

众所周知，文心一言是百度研发的知识增强大语言模型，能够与人对话互动，回答问题，协助创作，高效便捷地帮助人们获取信息、知识和灵感。

大家通过`ErnieBot-SDK`可以调用文心一言模型完成简单的AI任务，但是如果面对复杂的场景应用，可能需要基于`ErnieBot-SDK`开发较多的功能模块。

为了降低使用门槛和开发工作，我们推荐通过`ErnieBot-Agent`来调用文心一言模型，助力大家快速开发出AI应用。



## 1 准备

使用pip快速安装`ErnieBot-Agent`，要求`Python>=3.8`。

```shell
pip install --upgrade erniebot-agent
```

在使用`ErnieBot-Agent`之前，大家需要完成鉴权步骤：
* 在[AI Studio星河社区](https://aistudio.baidu.com/index)注册并登录账号
* 在个人中心的[访问令牌页面](https://aistudio.baidu.com/index/accessToken)获取用户凭证`Access Token`
* 通过环境变量或者`Python`代码设置`Access Token`

```shell
export EB_AGENT_ACCESS_TOKEN="your access token"
```

```python
import os
os.environ["EB_AGENT_ACCESS_TOKEN"] = "your access token"
```

## 2 Message使用指南

大家在使用文心一言模型之前，有必要先了解`Message`。

在`ErnieBot-Agent`中，文心一言模型通过`Message`和外界（用户或者其他模块）进行交互。具体而言，模型的输入是一个`Message`数组，输出是单条`Message`。

在`ErnieBot-Agent`中，主要有如下4类`Message`：
* `HumanMessage`：用户输入给模型的普通信息，比如聊天的问题。
* `SystemMessage`：用户输入给模型的全局信息，比如角色扮演的指令、输出格式设置的指令，通常一个`Message`数组中只有一条`SystemMessage`。
* `AIMessage`：模型返回的信息，比如聊天的回答、触发`Function call`的回答。
* `FunctionMessage`：上一轮模型的输出是带有`Funciton call`的`AIMessage`，则用户需要首先调用`Function`，然后将`Function`的结果输入给大语言模型。

创建`Message`的示例代码如下：

```python
import json
from erniebot_agent.memory import HumanMessage, SystemMessage, FunctionMessage

human_message = HumanMessage(content='你好，你是谁？')

system_message = SystemMessage(content='你是一个知识渊博的数学老师，使用浅显易懂的方法来回答问题')

result = {"temperature": 25, "unit": "摄氏度"}
function_message = FunctionMessage(name='get_current_temperature', content=json.dumps(result, ensure_ascii=False))

print(human_message)
print(system_message)
print(function_message)
```

示例的输出如下：
```
<role: 'user', content: '你好，你是谁？'>
<role: 'system', content: '你是一个知识渊博的数学老师，使用浅显易懂的方法来回答问题', token_count: 28>
<role: 'function', name: 'get_current_temperature', content: '{"temperature": 25, "unit": "摄氏度"}'>
```

## 3 ChatModels使用指南

出于使用场景和性能的考虑，`ErnieBot-Agent`只提供异步接口来使用文心一言模型，具体支持`ernie-3.5`、`ernie-turbo`、`ernie-4.0`和`ernie-longtext`。


### 3.1 进行文本补全

通过环境变量设置`Access Token`后，大家可以执行如下示例代码。

这个示例中，两次调用`async_chat`接口传入只有单条`HumanMessage`的数组，文心一言模型会对单条`HumanMessage`做出回答，返回一条`AIMessage`。

```python
import asyncio
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.memory import HumanMessage

async def demo():
    model = ERNIEBot(model="ernie-3.5")
    human_message = HumanMessage(content='你好，你是谁')
    ai_message = await model.async_chat(messages=[human_message])
    print(ai_message.content, '\n')

    human_message = HumanMessage(content='推荐三个深圳有名的景点')
    ai_message = await model.async_chat(messages=[human_message], stream=True)  # 流式返回
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
import asyncio
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.memory import HumanMessage

async def demo():
    model = ERNIEBot(model="ernie-3.5")
    messages = []

    messages.append(HumanMessage(content='推荐三个深圳有名的景点'))
    ai_message = await model.async_chat(messages=messages)
    messages.append(ai_message)
    print(ai_message.content, '\n')

    messages.append(HumanMessage(content='根据你推荐的景点，帮我做一份一日游的攻略'))
    ai_message = await model.async_chat(messages=messages)
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
