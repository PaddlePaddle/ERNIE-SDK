<div align="center">

<h1>ERNIE Bot Agent</h1>

`ERNIE Bot Agent` 是由百度飞桨全新推出的大模型智能体(agent)开发框架。基于文心大模型的编排能力，我们依托飞桨星河社区提供了丰富的预置平台化功能，并允许高度定制化的开发，旨在为开发者打造一站式的大模型Agent和应用搭建框架和平台。

[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/ERNIE-Bot-SDK.svg)](https://github.com/PaddlePaddle/ERNIE-Bot-SDK/releases)
![Supported Python versions](https://img.shields.io/badge/python-3.8+-orange.svg)
![Supported OSs](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)
[![Downloads](https://pepy.tech/badge/erniebot-agent)](https://pepy.tech/project/erniebot-agent)
[![codecov](https://codecov.io/gh/PaddlePaddle/ERNIE-Bot-SDK/branch/master/graph/badge.svg)](https://codecov.io/gh/PaddlePaddle/ERNIE-Bot-SDK)

</div>

![eb_sdk_agent_structure](https://github.com/PaddlePaddle/ERNIE-Bot-SDK/assets/11987277/6f62f191-fc7e-44ed-85f8-f7bcc210bcbb)

## 特性

### 强大的编排能力

与目前业界主流的通过prompt和output parser实现agent的方式不同，`ERNIE Bot Agent` 基于文心大模型的function calling实现了多工具编排和自动调度能力，并且允许工具、插件、知识库等不同组件的混合编排。除了自动调度，我们未来还将支持更多的编排模式，例如手动编排、半自动编排，为开发者提供更大的灵活性。

### 丰富的组件库

`ERNIE Bot Agent`为开发者提供了一个丰富的预置组件库：

- **预置工具**：只需一行代码，即可加载使用星河社区工具中心的30+预置工具。这些工具当前主要来自百度AI开发平台和飞桨特色PP系列模型。后续，我们会持续接入更多预置工具，也欢迎社区贡献。此外，工具模块也支持用户灵活自定义本地和远程工具。
- **知识库**：我们提供了开箱即用的基于文心百中的平台化知识库, 并允许开发者在二次开发的场景下使用LangChain、llama-index等主流开源库作为知识库。
- **文心一言插件**：我们将会支持通过`ERNIE Bot Agent`调用文心一言插件商城中的插件（开发中）

### 低开发门槛

我们希望能够降低开发门槛，使更多的开发者能够轻松构建智能体应用：

- **零代码界面**：依托星河社区，我们提供了零代码界面的智能体构建工具，通过简单的点击配置即可开发AI原生应用。
- **简洁的代码**：只需10行代码就可以快速开发一个智能体应用。
- **预置资源与平台支持**：大量的预置工具、平台级别的知识库，以及后续将推出的平台级别的记忆机制，都旨在加速开发过程。


## 快速安装

<details>
<summary>点击展开</summary>
建议您可以使用pip快速安装 `ERNIE Bot Agent` 的最新稳定版。

```shell
pip install --upgrade erniebot-agent
```

如需使用develop版本，可以下载源码后执行如下命令安装

```shell
git clone https://github.com/PaddlePaddle/ERNIE-Bot-SDK.git
cd ERNIE-Bot-SDK
pip install erniebot-agent
```
</details>



## 快速体验

<details>
<summary>点击展开</summary>

```python
# Todo: 添加快速体验代码
from erniebot_agent.chat_models import ERNIEBot
```

</details>

## ERNIE Bot SDK

`ERNIE Bot SDK` 作为 `ERNIE Bot Agent` 的底层依赖，为开发者提供了便捷易用的接口，使其能够轻松调用文心大模型的强大功能，涵盖了文本创作、通用对话、语义向量以及AI作图等多个方面。有关更多详细的使用指南，请参阅[ERNIE Bot SDK](.erniebot/README.md)

## License

ERNIE Bot Agent 和 ERNIE Bot SDK 遵循Apache-2.0开源协议。
