# Agent

## 1 Agent简介

在ERNIE Bot Agent框架中，agent是一个可以通过行动自主完成设定的目标的智能体。Agent具备自主理解、规划决策能力，能够执行复杂的任务。在用户与agent的每一轮交互中，agent接收一段自然语言文本作为输入，从输入中分析用户需求，确定需要完成的任务，通过调用外部工具等手段完成任务，最终产生一段自然语言书写的回复。Agent的能力来源于其集成的chat model、tool以及memory等组件。例如：chat model帮助agent理解和决策，tool可用于调用外部API，memory则赋予agent存储对话上下文的能力。ERNIE Bot Agent框架预置了一系列开箱即用的agent，同时也支持开发者根据需要定制自己的agent。

## 2 使用预置Agent

在阅读本节前，请首先熟悉chat model、tool、memory等组件的用法和注意事项。此外，本节的所有示例代码均需要在异步环境中执行。例如，可以使用如下方式编写Python脚本执行示例代码：

```python
import asyncio

async def main():
    # 将示例代码拷贝到这里


if __name__ == "__main__":
    asyncio.run(main())
```

关于agent相关类的详细接口，请参考[API文档](../package/erniebot_agent/agents.md)

#### 2.1 基础概念

为了更清晰地描述agent的运行机制，在介绍具体的agent之前，首先定义如下概念：

- **运行（run）**：指agent与用户的一轮交互。ERNIE Bot Agent框架中提供的agent均可配备memory组件，因此通常agent可以与用户进行多轮交互，多次运行之间的对话上下文能够被保存。
- **行动（action）**：指agent在运行中被规划执行的一个动作，例如调用tool。
- **步骤（step）**：指agent在运行中实际执行一次行动并得到结果。

#### 2.2 Function Agent

Function agent是一种由大语言模型的函数调用能力驱动的agent，这类agent与用户的一轮完整的交互流程如下：

1. Agent将用户输入包装为message，发送给chat model。
2. Chat model分析用户需求，为agent规划一个行动。chat model也可能认为agent无需执行行动，此时chat model将返回一段自然语言文本作为回复。
3. 如果chat model认为agent需要执行行动，并且当前步骤数未超过预先设定的最大限制，则agent根据chat model返回的信息执行相应的行动，然后将执行结果包装为message，再次发送给chat model，完成一个步骤。之后，回到步骤2。
4. 如果chat model认为agent无需执行行动，则agent完成本次运行，将chat model的回复返回给用户；如果步骤数量超过上限，则agent也完成本次运行，此时agent根据chat model返回的信息构造回复返回给用户。

以下是使用function agent的一些例子：

- 构造不配备有tool的function agent，用其进行多轮对话：

```python
from erniebot_agent.agents import FunctionAgent
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.memory import WholeMemory

agent = FunctionAgent(llm=ERNIEBot(model="ernie-3.5"), tools=[], memory=WholeMemory())

response = await agent.run("你好，小度！")
# `text`属性是agent给出的回复文本
print(response.text)
# 打印结果可能如下：
# 你好，有什么我可以帮你的吗？

response = await agent.run("我刚刚怎么称呼你？")
# `chat_history`属性存储本次运行中与chat model的对话历史
for message in response.chat_history:
    print(message.content)
# 打印结果可能如下：
# 我刚刚怎么称呼你？
# 您叫我小度。如果您有任何问题或需要帮助，请随时告诉我。
```

- 使用function agent调用tool完成用户给定的任务：

```python
from erniebot_agent.agents import FunctionAgent
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.tools.calculator_tool import CalculatorTool

# `Calculator`工具用于完成数学计算
# 如果没有传递`memory`参数，则agent默认构造和使用一个`WholeMemory`对象
agent = FunctionAgent(llm=ERNIEBot(model="ernie-3.5"), tools=[CalculatorTool()])

response = await agent.run("请问四加5*捌的结果是多少？")

print(response.text)
# 打印结果可能如下：
# 根据您的公式4+5*8，结果是44。如果您还有其他问题或需要计算其他公式，请随时告诉我。

# `steps`属性存储本次运行的所有步骤
print(response.steps)
# 打印结果可能如下：
# [ToolStep(info={'tool_name': 'CalculatorTool', 'tool_args': '{"math_formula":"4+5*8"}'}, result='{"formula_result": 44}', input_files=[], output_files=[])]

step = response.steps[0]
print("调用的tool名称：", step.info["tool_name"])
print("调用tool输入的参数（JSON格式）：", step.info["tool_args"])
print("调用tool返回的结果（JSON格式）：", step.result)
```

- 使用function agent编排多tool：

```python
from erniebot_agent.agents import FunctionAgent
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.tools.calculator_tool import CalculatorTool
from erniebot_agent.tools.current_time_tool import CurrentTimeTool

# 指定`enable_multi_step_tool_call`为True以启用多步工具调用功能
chat_model = ERNIEBot(model="ernie-3.5", enable_multi_step_tool_call=True)
agent = FunctionAgent(llm=chat_model, tools=[CalculatorTool()])

# 除了在构造agent时传入tool，还可以通过`load_tool`方法加载tool
# `CurrentTimeTool`工具用于获取当前时间
agent.load_tool(CurrentTimeTool())
# 与`load_tool`相对，`unload_tool`方法可用于卸载tool

# `get_tools`方法返回当前agent可以使用的所有tool
print(agent.get_tools())
# 打印结果如下：
# [<name: CalculatorTool, description: CalculatorTool用于执行数学公式计算>, <name: CurrentTimeTool, description: CurrentTimeTool 用于获取当前时间>]

response = await agent.run("请将当前时刻的时、分、秒数字相加，告诉我结果。")

print(response.text)
# 打印结果可能如下：
# 根据当前时间，时、分、秒数字相加的结果是68。

# 观察agent是否正确调用tool完成任务
for step in response.steps:
    print(step)
# 打印结果可能如下：
# ToolStep(info={'tool_name': 'CurrentTimeTool', 'tool_args': '{}'}, result='{"current_time": "2023年12月27日 21时39分08秒"}', input_files=[], output_files=[])
# ToolStep(info={'tool_name': 'CalculatorTool', 'tool_args': '{"math_formula":"21+39+8"}'}, result='{"formula_result": 68}', input_files=[], output_files=[])
```

- 使用function agent调用输入、输出中包含文件的tool：

```python
import aiohttp

from erniebot_agent.agents import FunctionAgent
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.file import GlobalFileManagerHandler
from erniebot_agent.tools import RemoteToolkit

# 获取PP-OCRv4工具箱与语音合成工具箱，并将其中的所有工具装配给agent
ocr_toolkit = RemoteToolkit.from_aistudio("pp-ocrv4")
tts_toolkit = RemoteToolkit.from_aistudio("texttospeech")
agent = FunctionAgent(llm=ERNIEBot(model="ernie-3.5"), tools=[*ocr_toolkit.get_tools(), *tts_toolkit.get_tools()])

# 下载示例图片
async with aiohttp.ClientSession() as session:
    async with session.get("https://paddlenlp.bj.bcebos.com/ebagent/ci/fixtures/remote-tools/ocr_example_input.png") as response:
        with open("example.png", "wb") as f:
            f.write(await response.read())

# 获取file manager，并使用其创建file
file_manager = GlobalFileManagerHandler().get()
input_file = await file_manager.create_file_from_path("example.png")

# 通过`files`参数在agent运行输入中包含file
response = await agent.run("请识别这张图片中的文字。", files=[input_file])

print(response.text)
# 打印结果可能如下：
# 根据您提供的图片，PP-OCRv4模型识别出了其中的文字，它们是“中国”和“汉字”。如果您需要更深入的分析或有其他问题，请随时告诉我。

assert len(response.steps) == 1
# `input_files`属性包含步骤涉及的所有输入文件
print(response.steps[0].input_files)
# 打印结果可能如下：
# [<LocalFile id: 'file-local-74aaf9e4-a4c2-11ee-b0a2-fa2020087eb4', filename: 'example.png', byte_size: 17663, created_at: '2023-12-27 22:15:58', purpose: 'assistants', metadata: {}, path: PosixPath('example.png')>]

# 尝试调用语音合成工具，该工具的输出中包含文件
response = await agent.run("请使用将刚才识别出的文字转换为语音。")

print(response.text)
# 打印结果可能如下：
# 根据您的需求，我为您合成了语音文件：file-local-4bab0eca-a4c3-11ee-a16e-fa2020087eb4。如果您需要进一步操作或有其他问题，请随时告诉我。

assert len(response.steps) == 1
# `output_files`属性包含步骤涉及的所有输出文件
output_files = response.steps[0].output_files
print(output_files)
# 打印结果可能如下：
# [<LocalFile id: 'file-local-4bab0eca-a4c3-11ee-a16e-fa2020087eb4', filename: 'tool-c3f5343c-6b33-4ee7-be31-35a60848bbd3.wav', byte_size: 43938, created_at: '2023-12-27 22:15:59', purpose: 'assistants_output', metadata: {'tool_name': 'texttospeech/v1.6/tts'}, path: PosixPath('/tmp/tmpd_ux8_ud/tool-c3f5343c-6b33-4ee7-be31-35a60848bbd3.wav')>]

# 将输出文件内容存储到指定文件中
assert len(output_files) == 1
await output_files[0].write_contents_to("output.wav")
```

#### 2.3 回调函数

为了使扩展agent功能更加便利，ERNIE Bot Agent框架支持为`erniebot_agent.agents.Agent`的子类装配回调函数。具体而言，在初始化agent时可以传入`callbacks`参数，以使agent在特定**事件**发生时调用相应的回调函数。当未指定`callbacks`参数或者将其设置为`None`时，agent将使用默认的回调函数。

#### 2.3.1 事件一览

RNIEBot-Agent框架定义了以下事件：

- `run_start`：Agent的运行开始。
- `llm_start`：Agent与chat model的交互开始。
- `llm_end`：Agent与chat model的交互成功结束。
- `llm_error`：Agent与chat model的交互发生错误。
- `tool_start`：Agent对tool的调用开始。
- `tool_end`：Agent对tool的调用成功结束。
- `tool_error`：Agent对tool的调用发生错误。
- `run_error`：Agent的运行发生错误。
- `run_end`：Agent的运行成功结束。

#### 2.3.2 默认回调函数

Agent默认装配的回调函数如下：

- `erniebot_agent.agents.callback.LoggingHandler`：日志记录回调函数集合。

#### 2.3.3 自定义回调函数

当默认回调函数无法满足需求时，可以通过继承基类`erniebot_agent.agents.callback.CallbackHandler`定制回调函数。具体而言，`erniebot_agent.agents.callback.CallbackHandler`提供一系列名为`on_{事件名称}`的方法，通过重写这些方法可以在特定事件发生时执行自定义逻辑。一个例子如下：

```python
from erniebot_agent.agents.callback import CallbackHandler

class CustomCallbackHandler(CallbackHandler):
    async def on_run_start(self, agent, prompt):
        print("Agent开始运行")

    async def on_run_end(self, agent, response):
        print("Agent结束运行，响应为：", response)
```

以上定义的`CustomCallbackHandler`在agent开始运行和结束运行时打印信息。


## 3 定制Agent

在部分情况下，预置的agent可能无法满足需求。为此，ERNIE Bot Agent框架也为用户提供定制agent的手段。在大部分情况下，推荐通过继承基类`erniebot_agent.agents.Agent`定制agent。通常，`erniebot_agent.agents.Agent`的子类只需要实现`_run`方法，开发者需要在其中实现自定义逻辑。

示例如下：

```python
from erniebot_agent.agents import Agent
from erniebot_agent.agents.schema import AgentResponse
from erniebot_agent.memory.messages import HumanMessage

class CustomAgent(Agent):
    async def _run(self, prompt, files=None):
        # `chat_history`与`steps_taken`分别用于记录本次运行的对话历史和步骤信息
        chat_history = []
        steps_taken = []

        # 构建输入message
        input_message = await HumanMessage.create_with_files(prompt, files or [])
        chat_history.append(input_message)
        # 将输入消息存储到memory中
        self.memory.add_message(input_message)

        # 与chat model交互
        llm_resp = await self.run_llm(self.memory.get_messages())
        chat_history.append(llm_resp.message)
        # 将输出消息存储到memory中
        self.memory.add_message(llm_resp.message)

        # 根据chat model的输出，决定是否执行行动
        # 如果需要执行行动，还需确定执行行动所需信息
        # 例如：用`should_run_tool`指示是否应该调用tool，在`action`中包含tool名称和输入参数
        ...

        if should_run_tool:
            # 调用tool
            tool_resp = await self.run_tool(action["tool_name"], action["tool_args"])
            # 将`tool_resp`转换为`erniebot_agent.agents.schema.ToolStep`对象`tool_step`
            ...
            steps_taken.append(tool_step)
            # 假设当前agent只执行至多一次行动，此时构造`AgentResponse`并返回
            # 如果agent已经完成任务，将`status`设置为"FINISHED"
            # 如果agent尚未完成任务，本次运行被提前终止，则将`status`设置为"STOPPED"
            return AgentResponse(
                text=self.memory.get_messages()[-1].content,
                chat_history=chat_history,
                steps=steps_taken,
                status="FINISHED",
            )

        # 其它处理逻辑
        ...
```