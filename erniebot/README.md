<div align="center">

<h1>ERNIE Bot</h1>

ERNIE Bot提供便捷易用的接口，可以调用文心大模型的能力，包含文本创作、通用对话、语义向量、AI作图等。

[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/ERNIE-SDK.svg)](https://github.com/PaddlePaddle/ERNIE-SDK/releases)
![Supported Python versions](https://img.shields.io/badge/python-3.8+-orange.svg)
![Supported OSs](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)

</div>

## 最新动态

* 支持文心一言ernie-bot-8k模型，输入token数量上限提升至7000。
* 支持基于文心大模型4.0版本的文心一言ernie-bot-4，用于对话补全任务。
* 增加`EBResponse.get_result`方法，支持从响应中取出关键信息。

## 教程与文档

请阅读[ERNIE Bot文档](https://ernie-bot-agent.readthedocs.io/zh-cn/latest/sdk/)。

## 快速安装

执行如下命令，快速安装最新版本的ERNIE Bot（要求Python >= 3.8)。

```shell
pip install --upgrade erniebot
```

## 快速使用

使用ERNIE Bot之前，请首先申请用户凭证并设置鉴权参数。

### Python接口

```python
import erniebot

# List supported models
models = erniebot.Model.list()

print(models)
# ernie-3.5               文心大模型（ernie-3.5）
# ernie-3.5-8k            文心大模型（ernie-3.5-8k）
# ernie-lite              文心大模型（ernie-lite）
# ernie-4.0               文心大模型（ernie-4.0）
# ernie-speed             文心大模型（ernie-speed）
# ernie-speed-128k        文心大模型（ernie-speed-128k）
# ernie-tiny-8k           文心大模型（ernie-tiny-8k）
# ernie-char-8k           文心大模型（ernie-char-8k）
# ernie-text-embedding    文心百中语义模型
# ernie-vilg-v2           文心一格模型

# Set authentication params
erniebot.api_type = "aistudio"
erniebot.access_token = "<access-token-for-aistudio>"

# Create a chat completion
response = erniebot.ChatCompletion.create(model="ernie-3.5", messages=[{"role": "user", "content": "你好，请介绍下你自己"}])

print(response.get_result())
```

### 命令行接口（CLI）

```bash
# List supported models
erniebot api model.list

# Set authentication params for chat_completion.create
export EB_API_TYPE="aistudio"
export EB_ACCESS_TOKEN="<access-token-for-aistudio>"

# Create a chat completion (using ernie-3.5, ernie-turbo, etc.)
erniebot api chat_completion.create --model ernie-3.5 --message user "请介绍下你自己"

# Set authentication params for image.create
export EB_API_TYPE="yinian"
export EB_ACCESS_TOKEN="<access-token-for-yinian>"

# Generate images via ERNIE-ViLG
erniebot api image.create --model ernie-vilg-v2 --prompt "画一只驴肉火烧" --height 1024 --width 1024 --image-num 1
```

## 典型示例

### 对话补全（Chat Completion）

ERNIE Bot提供具备对话补全能力的ernie-3.5、ernie-turbo、ernie-4.0、ernie-3.5-8k等文心大模型。

不同模型在效果、速度等方面各有千秋，大家可以根据实际场景的需求选择合适的模型。

以下是调用ernie-3.5模型进行多轮对话的示例：

```python
import erniebot

erniebot.api_type = "aistudio"
erniebot.access_token = "<access-token-for-aistudio>"

response = erniebot.ChatCompletion.create(
    model="ernie-3.5",
    messages=[{
        "role": "user",
        "content": "请问你是谁？"
    }, {
        "role": "assistant",
        "content":
        "我是百度公司开发的人工智能语言模型，我的中文名是文心一言，英文名是ERNIE-Bot，可以协助您完成范围广泛的任务并提供有关各种主题的信息，比如回答问题，提供定义和解释及建议。如果您有任何问题，请随时向我提问。"
    }, {
        "role": "user",
        "content": "我在深圳，周末可以去哪里玩？"
    }])

print(response.get_result())
```

### 语义向量（Embedding）

ERNIE Bot提供用于生成文本的向量表示的ernie-text-embedding模型。

该模型基于文心大模型，使用海量数据训练得到，为[文心百中](https://wenxin.baidu.com/baizhong/index/)系统提供关键能力。该模型可以将输入文本转为384维浮点数表示的向量。这些向量具备精准的语义表达能力，可用于度量文本之间的语义相似度。

大家可以使用以下代码生成句子的向量表示：

```python
import erniebot

erniebot.api_type = "aistudio"
erniebot.access_token = "<access-token-for-aistudio>"

response = erniebot.Embedding.create(
    model="ernie-text-embedding",
    input=[
        "我是百度公司开发的人工智能语言模型，我的中文名是文心一言，英文名是ERNIE-Bot，可以协助您完成范围广泛的任务并提供有关各种主题的信息，比如回答问题，提供定义和解释及建议。如果您有任何问题，请随时向我提问。",
        "2018年深圳市各区GDP"
        ])

print(response.get_result())
```

大家可以登陆[文心百中体验中心](https://wenxin.baidu.com/baizhong/knowledgesearch)，体验更多大模型语义搜索的能力。

### 文生图（Image Generation）

ERNIE Bot提供具备文生图能力的ernie-vilg-v2大模型。

该模型具备丰富的风格与强大的中文理解能力，支持生成多种尺寸的图片。

```python
import erniebot

erniebot.api_type = "yinian"
erniebot.access_token = "<access-token-for-yinian>"

response = erniebot.Image.create(
    model="ernie-vilg-v2",
    prompt="雨后的桃花，8k，辛烷值渲染",
    width=512,
    height=512
)

print(response.get_result())
```

<img width="512" src="https://github.com/PaddlePaddle/ERNIE-SDK/assets/1371212/73911c97-ef42-4803-8dc6-d385486c128c" />

我们推荐两个撰写文生图提示词的文档，大家可以组合使用，创作出更加精美的图片。

* [AI作画-基础版使用指南](https://ai.baidu.com/ai-doc/NLP/qlakgh129)
* [AI作画-高级版使用指南](https://ai.baidu.com/ai-doc/NLP/4libyluzs)

大家也可登陆[文心一格](https://yige.baidu.com/)平台，体验更多AI艺术与创意辅助的能力。

### 函数调用（Function Calling）

ERNIE Bot提供函数调用功能，即由大模型根据对话上下文确定何时以及如何调用函数。

借由函数调用，用户可以从大模型获取结构化数据，进而利用编程手段将大模型与已有的内外部API结合以构建应用。

以下是调用ernie-3.5模型进行函数调用的示例：

```python
import erniebot

erniebot.api_type = "aistudio"
erniebot.access_token = "<access-token-for-aistudio>"

response = erniebot.ChatCompletion.create(
    model="ernie-3.5",
    messages=[{
        "role": "user",
        "content": "深圳市今天气温多少摄氏度？",
    }, ],
    functions=[
        {
            "name": "get_current_temperature",
            "description": "获取指定城市的气温",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名称",
                    },
                    "unit": {
                        "type": "string",
                        "enum": [
                            "摄氏度",
                            "华氏度",
                        ],
                    },
                },
                "required": [
                    "location",
                    "unit",
                ],
            },
            "responses": {
                "type": "object",
                "properties": {
                    "temperature": {
                        "type": "integer",
                        "description": "城市气温",
                    },
                    "unit": {
                        "type": "string",
                        "enum": [
                            "摄氏度",
                            "华氏度",
                        ],
                    },
                },
            },
        },
    ],
    stream=False)
print(response.get_result())
```

## Gradio Demos

为了让用户更全面、更直观地了解ERNIE Bot的各项功能，我们基于Gradio开发了一系列带有web用户界面的演示应用。请参阅[说明文档](./examples/README.md)，尝试对话补全、语义向量、文生图、函数调用等可交互例子。

<img width="1296" src="https://user-images.githubusercontent.com/19339784/264367116-600c34b9-0103-4fb7-bbe5-6d71ddc6af09.gif" />

## Acknowledgement

我们借鉴了[OpenAI Python Library](https://github.com/openai/openai-python)的部分API设计，在此对OpenAI Python Library作者及开源社区表示感谢。

## Community Contribution

- 感谢[@chenwanqq](https://github.com/chenwanqq)贡献Rust版本的[erniebot-rs](https://github.com/chenwanqq/erniebot-rs)
