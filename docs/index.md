---
hide:
  - navigation
  - toc
---

<div align="center">

<h1>ERNIE Bot SDK</h1>

</div>

ERNIE Bot SDK 仓库包含两个项目：ERNIE Bot Agent 和 ERNIE Bot。ERNIE Bot Agent 是百度飞桨推出的基于文心大模型编排能力的大模型智能体开发框架，结合了飞桨星河社区的丰富预置平台功能。ERNIE Bot 则为开发者提供便捷接口，轻松调用文心大模型的文本创作、通用对话、语义向量及AI作图等基础功能。

![eb_sdk_agent_structure](https://github.com/PaddlePaddle/ERNIE-Bot-SDK/assets/11987277/1fbcfbca-7695-4cca-9b4f-35a49d1d7c52)


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

## ERNIE Bot

**ERNIE Bot** 作为 **ERNIE Bot Agent** 的底层依赖，为开发者提供了便捷易用的接口，使其能够轻松调用文心大模型的强大功能，涵盖了文本创作、通用对话、语义向量以及AI作图等多个基础功能。有关更多详细的使用指南，请参阅[ERNIE Bot](./sdk/README.md)。

## License

ERNIE Bot Agent 和 ERNIE Bot 遵循Apache-2.0开源协议。
