# 快速开始

下面的`quick_start.py`示例展示了如何使用 ERNIE Bot Agent 快速构建智能体应用。

```python
import asyncio
import os

from erniebot_agent.agents import FunctionAgent
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.tools import RemoteToolkit

async def main():
    llm = ERNIEBot(model="ernie-3.5")  # 初始化大语言模型
    # 这里以语音合成工具为例子，更多的预置工具可参考 https://aistudio.baidu.com/application/center/tool
    tts_tool = RemoteToolkit.from_aistudio("texttospeech").get_tools()
    agent = FunctionAgent(llm=llm, tools=tts_tool)  # 创建智能体，集成语言模型与工具

    # 与智能体进行通用对话
    result = await agent.run("你好，请自我介绍一下")
    print(f"Agent输出: {result.text}")

    # 请求智能体根据输入文本，自动调用语音合成工具
    result = await agent.run("把上一轮的自我介绍转成语音")
    print(f"Agent输出: {result.text}")

    # 将智能体输出的音频文件写入test.wav, 可以尝试播放
    audio_file = result.steps[-1].output_files[0]
    await audio_file.write_contents_to("./test.wav")

asyncio.run(main())
```

在运行上述代码之前，您需要先配置 `EB_AGENT_ACCESS_TOKEN` 到环境变量中。同时，建议您通过配置 `EB_AGENT_LOGGING_LEVEL` 开启info日志，以便查看更多智能体运行信息。

执行以下命令配置环境变量并运行示例代码：

```shell
export EB_AGENT_ACCESS_TOKEN=<aistudio-access-token>
export EB_AGENT_LOGGING_LEVEL=info
python quick_start.py
```

执行完毕后，您将在当前目录下生成一个名为test.wav的音频文件，该文件中包含智能体的语音自我介绍。您可以使用任何支持音频文件播放的设备或软件来播放该文件，以便更好地了解智能体的语音特征。此外，我们还可以从智能体打印的日志中获取更详细的执行信息，从而更好地了解整个执行过程。

```shell
INFO - [Run][Start] FunctionAgent is about to start running with input:
你好，请自我介绍一下
INFO - [LLM][Start] ERNIEBot is about to start running with input:
 role: user 
 content: 你好，请自我介绍一下 

INFO - [LLM][End] ERNIEBot finished running with output:
 role: assistant 
 content: 你好，我是文心一言，英文名是ERNIE Bot。我能够与人对话互动，回答问题，协助创作，高效便捷地帮助人们获取信息、知识和灵感。 
INFO - [Run][End] FunctionAgent finished running.
Agent输出：你好，我是文心一言，英文名是ERNIE Bot。我能够与人对话互动，回答问题，协助创作，高效便捷地帮助人们获取信息、知识和灵感。
INFO - [Run][Start] FunctionAgent is about to start running with input:
把上一轮的自我介绍转成语音
INFO - [LLM][Start] ERNIEBot is about to start running with input:
 role: user 
 content: 你好，请自我介绍一下 
 role: assistant 
 content: 你好，我是文心一言，英文名是ERNIE Bot。我能够与人对话互动，回答问题，协助创作，高效便捷地帮助人们获取信息、知识和灵感。 
 role: user 
 content: 把上一轮的自我介绍转成语音 

INFO - [LLM][End] ERNIEBot finished running with output:
 role: assistant 
 function_call: 
{
  "name": "texttospeech/v1.6/tts",
  "thoughts": "用户希望将自我介绍转成语音，我可以使用texttospeech工具来实现这个需求。",
  "arguments": "{\"tex\":\"你好，我是文心一言，英文名是ERNIE Bot。我能够与人对话互动，回答问题，协助创作，高效便捷地帮助人们获取信息、知识和灵感。\"}"
} 
INFO - [Tool][Start] RemoteTool is about to start running with input:
{
  "tex": "你好，我是文心一言，英文名是ERNIE Bot。我能够与人对话互动，回答问题，协助创作，高效便捷地帮助人们获取信息、知识和灵感。"
}
INFO - [Tool][End] RemoteTool finished running with output:
{
  "audio": "file-local-2dd726b2-a492-11ee-8c08-506b4b225bd6",
  "prompt": "参考工具说明中对各个结果字段的描述，提取工具调用结果中的信息，生成一段通顺的文本满足用户的需求。请务必确保每个符合'file-'格式的字段只出现一次，无需将其转换为链接，也无需添加任何HTML、Markdown或其他格式化元素。"
}
INFO - [LLM][Start] ERNIEBot is about to start running with input:
 role: user 
 content: 你好，请自我介绍一下 
 role: assistant 
 content: 你好，我是文心一言，英文名是ERNIE Bot。我能够与人对话互动，回答问题，协助创作，高效便捷地帮助人们获取信息、知识和灵感。 
 role: user 
 content: 把上一轮的自我介绍转成语音 
 role: assistant 
 function_call: 
{
  "name": "texttospeech/v1.6/tts",
  "thoughts": "用户希望将自我介绍转成语音，我可以使用texttospeech工具来实现这个需求。",
  "arguments": "{\"tex\":\"你好，我是文心一言，英文名是ERNIE Bot。我能够与人对话互动，回答问题，协助创作，高效便捷地帮助人们获取信息、知识和灵感。\"}"
} 
 role: function 
 name: texttospeech/v1.6/tts 
 content: {"audio": "file-local-2dd726b2-a492-11ee-8c08-506b4b225bd6", "prompt": "参考工具说明中对各个结果字段的描述，提取工具调用结果中的... 

INFO - [LLM][End] ERNIEBot finished running with output:
 role: assistant 
 content: 根据你的请求，我已经将自我介绍转成了语音文件，并保存在了本地，文件名为file-local-2dd726b2-a492-11ee-8c08-506b4b225bd6。你可以使用任何支持该格式的播放器进... 
INFO - [Run][End] FunctionAgent finished running.
Agent输出：根据你的请求，我已经将自我介绍转成了语音文件，并保存在了本地，文件名为file-local-2dd726b2-a492-11ee-8c08-506b4b225bd6。你可以使用任何支持该格式的播放器进行播放。如果你需要进一步操作或有其他问题，请随时告诉我。
```

从日志中，我们可以清晰地看到智能体在处理两个不同请求时的运作流程。在第一个请求中，智能体直接利用大语言模型ERNIEBot进行了自我介绍。而在面对第二个请求，即需要将自我介绍转化为语音时，智能体的操作则复杂一些。

首先，ERNIEBot判断出需要使用语音合成工具来实现这一需求，并自动填充了必要的请求参数。接着，ERNIE Bot Agent 框架便开始自动调度和运行语音合成工具。这一切都在后台顺利进行，无需人工干预。当语音合成工具运行完毕后，它将生成的结果交给了ERNIEBot进行最后的润色和处理。ERNIEBot结合之前的对话内容和工具的输出，生成了一段既符合语境又满足用户需求的回答。
