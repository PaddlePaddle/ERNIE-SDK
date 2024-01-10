<div align="center">

<h1>ERNIE SDK</h1>

[开发文档](http://ernie-bot-agent.readthedocs.io/) | [智能体应用体验](https://aistudio.baidu.com/application/center?tag=agent)

[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/ERNIE-SDK.svg)](https://github.com/PaddlePaddle/ERNIE-SDK/releases)
![Supported Python versions](https://img.shields.io/badge/python-3.8+-orange.svg)
![Supported OSs](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)
[![Downloads](https://pepy.tech/badge/erniebot-agent)](https://pepy.tech/project/erniebot-agent)
[![codecov](https://codecov.io/gh/PaddlePaddle/ERNIE-SDK/branch/develop/graph/badge.svg)](https://codecov.io/gh/PaddlePaddle/ERNIE-SDK)

</div>

ERNIE Bot SDK 仓库包含两个项目：ERNIE Bot Agent 和 ERNIE Bot。ERNIE Bot Agent 是百度飞桨推出的基于文心大模型编排能力的大模型智能体开发框架，结合了飞桨星河社区的丰富预置平台功能。ERNIE Bot 则为开发者提供便捷接口，轻松调用文心大模型的文本创作、通用对话、语义向量及AI作图等基础功能。

![eb_sdk_agent_structure](https://github.com/PaddlePaddle/ERNIE-SDK/assets/11987277/1fbcfbca-7695-4cca-9b4f-35a49d1d7c52)

## ERNIE Bot Agent

### 特性

- **编排能力**: ERNIE Bot Agent 基于文心大模型的 Function Calling 能力实现了多工具编排和自动调度功能，并且允许工具、插件、知识库等不同组件的混合编排。除了自动调度，我们未来还将支持更多的编排模式，例如手动编排、半自动编排，为开发者提供更大的灵活性。
- **组件库**: ERNIE Bot Agent 为开发者提供了一个丰富的预置组件库：
    - **预置工具**：只需一行代码，即可加载使用星河社区工具中心的30+预置工具。这些工具当前主要来自百度AI开发平台和飞桨特色PP系列模型。后续，我们会持续接入更多预置工具，也欢迎社区贡献。此外，工具模块也支持用户灵活自定义本地和远程工具。
    - **知识库**：提供了开箱即用的基于文心百中的平台化知识库, 并允许开发者在二次开发的场景下使用[langchain](https://github.com/langchain-ai/langchain)、[llama_index](https://github.com/run-llama/llama_index)等主流开源库作为知识库。
    - **文心一言插件**：未来将支持通过调用文心一言插件商城中的插件（开发中）
- **低开发门槛**
    - **零代码界面**：依托星河社区提供了零代码界面的智能体构建工具，通过简单的点击配置即可开发AI原生应用。
    - **简洁的代码**：10行代码就可以快速开发一个智能体应用。
    - **预置资源与平台支持**：大量的预置工具、平台级别的知识库，以及后续将推出的平台级别的记忆机制，都旨在加速开发过程。

### 安装

<details>
<summary>点击展开</summary>

#### 源码安装

执行如下命令，使用源码安装 ERNIE Bot Agent（要求Python >= 3.8)。

```shell
git clone https://github.com/PaddlePaddle/ERNIE-SDK.git
cd ERNIE-SDK

# 首先安装Ernie Bot
pip install ./erniebot

# 然后安装ERNIE Bot Agent
pip install ./erniebot-agent            # 安装核心模块
# pip install './erniebot-agent/.[all]'   # 也可以加上[all]一次性安装所有模块，包括gradio等依赖库
```

#### 快速安装

执行如下命令，快速安装最新版本 ERNIE Bot Agent（要求Python >= 3.8)。

```shell
# 安装核心模块
pip install --upgrade erniebot-agent

# 安装所有模块
pip install --upgrade erniebot-agent[all]
```
</details>

### 快速体验

<details>
<summary>点击展开</summary>

下面的`quick_start.py`示例展示了如何使用 ERNIE Bot Agent 快速构建智能体应用。

```python
import asyncio

from erniebot_agent.agents import FunctionAgent
from erniebot_agent.chat_models import ERNIEBot
from erniebot_agent.tools import RemoteToolkit

async def main():
    llm = ERNIEBot(model="ernie-3.5")  # 初始化大语言模型
    tts_tool = RemoteToolkit.from_aistudio("texttospeech").get_tools()  # 获取语音合成工具
    agent = FunctionAgent(llm=llm, tools=tts_tool)  # 创建智能体，集成语言模型与工具

    # 与智能体进行通用对话
    result = await agent.run("你好，请自我介绍一下")
    print(result.text)
    # 模型返回类似如下结果：
    # 你好，我叫文心一言，是百度研发的知识增强大语言模型，能够与人对话互动，回答问题，协助创作，高效便捷地帮助人们获取信息、知识和灵感。

    # 请求智能体根据输入文本，自动调用语音合成工具
    result = await agent.run("把上一轮的自我介绍转成语音")
    print(result.text)
    # 模型返回类似如下结果：
    # 根据你的请求，我已经将自我介绍转换为语音文件，文件名为file-local-c70878b4-a3f6-11ee-95d0-506b4b225bd6。
    # 你可以使用任何支持播放音频文件的设备或软件来播放这个文件。如果你需要进一步操作或有其他问题，请随时告诉我。

    # 将智能体输出的音频文件写入test.wav, 可以尝试播放
    audio_file = result.steps[-1].output_files[0]
    await audio_file.write_contents_to("./test.wav")

asyncio.run(main())
```

运行上述代码，大家首先需要在[AI Studio星河社区](https://aistudio.baidu.com/index)注册并登录账号，然后在AI Studio的[访问令牌页面](https://aistudio.baidu.com/index/accessToken)获取`Access Token`，最后执行以下命令:
```shell
export EB_AGENT_ACCESS_TOKEN=<aistudio-access-token>
export EB_AGENT_LOGGING_LEVEL=info
python quick_start.py
```
</details>

### 详细教程

教程[链接](https://ernie-bot-agent.readthedocs.io/zh-cn/latest/)。

## ERNIE Bot

ERNIE Bot 作为 ERNIE Bot Agent 的底层依赖，为开发者提供了便捷易用的接口，使其能够轻松调用文心大模型的强大功能，涵盖了文本创作、通用对话、语义向量以及AI作图等多个基础功能。

更多详细的使用指南，请参阅[ERNIE Bot](./erniebot/README.md)。

## License

ERNIE Bot Agent 和 ERNIE Bot 遵循Apache-2.0开源协议。
