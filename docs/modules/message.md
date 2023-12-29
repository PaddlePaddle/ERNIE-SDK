# Message模块

## 1. 简介

简而言之，`Message`是大语言模型的输入输出信息进行封装。大家可以使用`Message`和大语言模型进行交互，后续使用`Memory`模块也会接触`Message`。

在`ERNIE Bot Agent`中，主要有如下4类`Message`：

* `HumanMessage`：用户输入给模型的普通信息，比如聊天的问题。
* `SystemMessage`：用户输入给模型的全局信息，比如角色扮演的指令、输出格式设置的指令。
* `AIMessage`：模型返回的信息，比如聊天的回答、触发`Function call`的回答。
* `FunctionMessage`：上一轮模型的输出是带有`Funciton call`的`AIMessage`，则用户需要首先调用`Function`，然后将`Function`的结果输入给大语言模型。

## 2. 核心类

下面简单介绍`Message`模块的核心类，详细接口请参考[API文档](../package/erniebot_agent/message.md)。

`Message`类是基类，以下是主要的属性和方法。

| 属性       | 类型           | 描述                                                      |
| ---------- | -------------- | ------------------------------------------------------- |
| role         | str          | Message的角色，有四类：user, system, assistant和function   |
| content      | str          | Mesage的文本内容                                          |
| token_count  | int          | 文本内容的token长度                                        |

| 方法              | 描述                                                                  |
| ----------------- | -------------------------------------------------------------------- |
| to_dict           | 将Message中核心属性转成字典                                              |
| token_count       | 设置token_count的大小                                                  |

`SystemMessage`类继承`Message`类，以下是主要的属性。

| 属性       | 类型           | 描述                                                      |
| ---------- | -------------- | ------------------------------------------------------- |
| role         | str          | Message的角色，等于system                                 |
| content      | str          | Mesage的文本内容                                          |
| token_count  | int          | 文本内容的token长度，近似计算len(content)                   |

`HumanMessage`类继承`Message`类，以下是主要的属性和新增的方法。

| 属性       | 类型           | 描述                                                      |
| ---------- | -------------- | ------------------------------------------------------- |
| role         | str          | Message的角色，等于user                                   |
| content      | str          | Mesage的文本内容                                          |
| token_count  | int          | 文本内容的token长度                                        |

| 方法              | 描述                                                                  |
| ----------------- | -------------------------------------------------------------------- |
| create_with_files | 创建带有File的HumanMessage                                             |

`AIMessage`类继承`Message`类，以下是主要的属性。

| 属性       | 类型           | 描述                                                      |
| ---------- | -------------- | ------------------------------------------------------- |
| role         | str          | Message的角色，等于assistant                              |
| content      | str          | Mesage的文本内容                                          |
| token_count  | int          | 文本内容的token长度                                        |
| function_call | Optional[FunctionCall] | FunctionCall的schema描述信息                   |


`FunctionMessage`类继承`Message`类，以下是主要的属性。

| 属性       | 类型           | 描述                                                      |
| ---------- | -------------- | ------------------------------------------------------- |
| role         | str          | Message的角色，等于user                                   |
| content      | str          | Mesage的文本内容                                          |
| token_count  | int          | 文本内容的token长度                                        |

## 3. 使用示例

为了直观展示，我们举例进行说明，请先确保完成`ERNIE Bot Agent`的安装和鉴权步骤。

### 示例1

大家在使用`ERNIE Bot`调用文心一言进行多轮对话时，需要按照规范定义每轮对话的信息（如下），稍显复杂。

```python
import erniebot

erniebot.api_type = "aistudio"
erniebot.access_token = "<access-token-for-aistudio>"
messages = [{
                "role": "user",
                "content": "我在深圳，周末可以去哪里玩"
            }, {
                "role": "assistant",
                "content":"深圳有许多著名的景点，以下是三个推荐景点：1. 深圳世界之窗，2. 深圳欢乐谷，3. 深圳东部华侨城。"
            }, {
                "role": "user",
                "content": "从你推荐的三个景点中，选出最值得去的景点是什么，直接给出景点名字即可"
            }]
response = erniebot.ChatCompletion.create(
    model="ernie-3.5",
    messages=messages
    )

print(response.get_result())
```

如果基于`ERNIE Bot Agent`调用文心一言，大家使用`Message`模块，可以较好地简化代码。

```python
import os
import asyncio
from erniebot_agent.memory import HumanMessage, AIMessage
from erniebot_agent.chat_models import ERNIEBot

os.environ["EB_AGENT_ACCESS_TOKEN"] = "your access token"

async def demo():
    model = ERNIEBot(model="ernie-3.5")
    # 使用Message模块
    messages = [HumanMessage("我在深圳，周末可以去哪里玩"),
                AIMessage("深圳有许多著名的景点，以下是三个推荐景点：1. 深圳世界之窗，2. 深圳欢乐谷，3. 深圳东部华侨城。"),
                HumanMessage("从你推荐的三个景点中，选出最值得去的景点是什么，直接给出景点名字即可")]
    ai_message = await model.chat(messages=messages)
    print(ai_message.content)

asyncio.run(demo())
```

### 示例2

创建各种`Message`的示例代码如下：

```python
import json
from erniebot_agent.memory import HumanMessage, SystemMessage, FunctionMessage

human_message = HumanMessage(content='你好，你是谁？')

system_message = SystemMessage(content='你是一名数学老师，使用浅显易懂的方法来回答问题')

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


### 示例3

使用`SystemMessage`的示例代码如下：

```python
import os
import asyncio
from erniebot_agent.memory import HumanMessage, SystemMessage
from erniebot_agent.chat_models import ERNIEBot

os.environ["EB_AGENT_ACCESS_TOKEN"] = "your access token"

async def demo():
    model = ERNIEBot(model="ernie-3.5")
    system_message = SystemMessage(content="你是一名数学老师，尽量使用浅显易懂的方法来解答问题")
    messages = [HumanMessage("勾股定理是什么")]
    ai_message = await model.chat(messages=messages, system=system_message.content)
    print(ai_message.content)

asyncio.run(demo())
```