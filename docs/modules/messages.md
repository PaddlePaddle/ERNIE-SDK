# Message模块

## 1. Message简介

简而言之，`Message`是大语言模型的输入输出信息进行封装。大家可以使用`Message`和大语言模型进行交互，后续使用`Memory`模块也会接触`Message`。

在`EB-Agent`中，主要有如下4类`Message`：

* `HumanMessage`：用户输入给模型的普通信息，比如聊天的问题。
* `SystemMessage`：用户输入给模型的全局信息，比如角色扮演的指令、输出格式设置的指令，通常一个`Message`数组中只有一条`SystemMessage`。
* `AIMessage`：模型返回的信息，比如聊天的回答、触发`Function call`的回答。
* `FunctionMessage`：上一轮模型的输出是带有`Funciton call`的`AIMessage`，则用户需要首先调用`Function`，然后将`Function`的结果输入给大语言模型。

## 2. Message的使用示例

为了直观展示，我们举例进行说明，请先确保完成`EB-Agent`的安装和鉴权步骤。

### 示例1

大家在使用`ERNIE Bot SDK`调用文心一言进行多轮对话时，需要按照规范定义每轮对话的信息（如下），稍显复杂。

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

如果切换使用`EB-Agent`调用文心一言，大家使用`Message`模块，可以较好地简化代码。

```python
import os
import asyncio
from erniebot_agent.memory import HumanMessage, AIMessage
from erniebot_agent.chat_models import ERNIEBot

os.environ["EB_AGENT_ACCESS_TOKEN"] = "your access token"

async def demo():
    model = ERNIEBot(model="ernie-3.5")
    messages = [HumanMessage("我在深圳，周末可以去哪里玩"),
                AIMessage("深圳有许多著名的景点，以下是三个推荐景点：1. 深圳世界之窗，2. 深圳欢乐谷，3. 深圳东部华侨城。"),
                HumanMessage("从你推荐的三个景点中，选出最值得去的景点是什么，直接给出景点名字即可")]
    ai_message = await model.async_chat(messages=messages)
    print(ai_message.content)

asyncio.run(demo())
```

### 示例2

创建`Message`的示例代码：

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


## 3.0 Message的API接口

`Message`模块的API接口，请参考[文档](../../package/erniebot_agent/messages/)。