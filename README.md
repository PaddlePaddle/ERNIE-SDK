# ERNIE Bot SDK

ERNIE Bot SDK 提供一系列便捷易用的接口来调用文心大模型的能力，包含文本创作、通用对话、语义向量、AI作图等能力。

## 快速安装

执行如下命令，快速安装Python语言的ERNIE Bot SDK（推荐Python >= 3.8)。

```shell
pip install --upgrade erniebot
```

## 快速使用

大家使用ERNIE Bot SDK之前，请申请AK/SK进行认证鉴权，具体请参考[认证鉴权](./docs/authentication.md)。

### Python 接口

```python
import erniebot

# List supported models
models = erniebot.Model.list()

print(models)
# ernie-bot-3.5         文心一言旗舰版
# ernie-bot-turbo       文心一言轻量版
# ernie-text-embedding  文心百中语义模型
# ernie-vilg-v2         文心一格模型

# Set ak/sk 
erniebot.ak = "<EB-ACCESS-KEY>"
erniebot.sk = "<EB-SECRET-KEY>"

# Create a chat completion
chat_completion = erniebot.ChatCompletion.create(model="ernie-bot-3.5", messages=[{"role": "user", "content": "你好，请介绍下你自己"}])

print(chat_completion.result)
```

### 命令行接口 (CLI)

```bash
# List supported models
erniebot api model.list

# Set api_type, ak, sk for chat_completion.create
export EB_API_TYPE="qianfan"
export EB_AK="<EB-ACCESS-KEY>"
export EB_SK="<EB-SECRET-KEY>"

# Create a chat completion (ernie-bot-3.5, ernie-bot-turbo, etc.)
erniebot api chat_completion.create --model ernie-bot-3.5 --message user '请介绍下你自己'

# Set api_type, ak, sk for image.create
export EB_API_TYPE="yinian"
export EB_AK="<EB-ACCESS-KEY>"
export EB_SK="<EB-SECRET-KEY>"

# Generate images via ERNIE-ViLG
erniebot api image.create --model ernie-vilg-v2 --prompt '画一只驴肉火烧' --height 1024 --width 1024 --image-num 1
```

## 经典示例

### 对话补全 (Chat Completion)

ERNIE Bot SDK提供了对话补全能力的文心一言旗舰版模型`ernie-bot-3.5`和文心一言迅捷版模型`ernie-bot-turbo`。
旗舰版模型的效果更好，迅捷版模型的响应速度更快、推理时延更低，大家可以根据实际场景的需求选择合适的模型。

以下是调用文心一言旗舰版模型进行多轮对话的示例。

```python
import erniebot

erniebot.ak = "<EB-ACCESS-KEY>"
erniebot.sk = "<EB-SECRET-KEY>"

completion = erniebot.ChatCompletion.create(
    model='ernie-bot-3.5',
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

print(completion)
```


### 语义向量 (Embedding)

ERNIE Bot SDK提供了提取语义向量的Embedding模型。
该模型基于文心大模型，使用海量数据训练得到，为[文心百中](https://wenxin.baidu.com/baizhong/index/)系统提供关键能力。该模型可以将字符串转为384维浮点数表达的语义向量，语义向量具备极其精准的语义表达能力，可以用于度量两个字符串之间的语义相似度。

大家可以使用以下代码提取句子的语义向量。

```python
import erniebot

erniebot.ak = "<EB-ACCESS-KEY>"
erniebot.sk = "<EB-SECRET-KEY>"

embedding = erniebot.Embedding.create(
    model="ernie-text-embedding",
    input=[
        "我是百度公司开发的人工智能语言模型，我的中文名是文心一言，英文名是ERNIE-Bot，可以协助您完成范围广泛的任务并提供有关各种主题的信息，比如回答问题，提供定义和解释及建议。如果您有任何问题，请随时向我提问。",
        "2018年深圳市各区GDP"
        ])

print(embedding)
```

大家可以登陆[文心百中体验中心](https://wenxin.baidu.com/baizhong/knowledgesearch)，体验更多大模型语义搜索的能力。

### 文生图（Image Generation）

ERNIE Bot SDK提供了文生图能力的ERNIE-ViLG大模型。
该模型具备丰富的风格与强大的中文理解能力，支持生成多种尺寸的图片。

```python
import erniebot

erniebot.ak = "<EB-ACCESS-KEY>"
erniebot.sk = "<EB-SECRET-KEY>"
erniebot.api_type = "yinian"

response = erniebot.Image.create(
    model="ernie-vilg-v2",
    prompt="画一个胸有成竹的男人",
    width=512,
    height=512
)

print(response)

```

<img width="624" alt="image" src="https://github.com/PaddlePaddle/ERNIE-Bot-sdk/assets/1371212/cad754ed-ed9d-42d0-a0dc-9ac29f04f67b">

我们推荐两个撰写文生图Prompt提示词的文档，大家可以组合使用，创作出更加精美的图片。
* [AI作画-基础版使用指南](https://ai.baidu.com/ai-doc/NLP/qlakgh129)
* [AI作画-高级版使用指南](https://ai.baidu.com/ai-doc/NLP/4libyluzs)

大家也可登陆[文心一格](https://yige.baidu.com/)平台，体验更多AI艺术与创意辅助的能力。

## Gradio可视化应用

为了让开发者可以更全面更低门槛的了解ERNIE Bot SDK的全功能，我们基于Gradio实现了一个功能丰富的可视化界面，参阅[示例说明](examples)可以快速本地测试ChatCompletion、Embedding和Image的可视化交互示例。

<img width="1296" alt="36dd85dbe30682a287b6a5c5d13e0cdc" src="https://github.com/PaddlePaddle/ERNIE-Bot-sdk/assets/1371212/17cfe2fa-efe0-4c06-8aad-3363d7b6cc40">


## 完整教程文档

* 快速开始
  * [安装](./docs/installation.md)
  * [认证鉴权](./docs/authentication.md)
  * [参数配置](./docs/configuration.md)
* 使用指南
  * [对话补全ChatCompletion](./docs/guide/chat_completion.md)
  * [语义向量Embedding](./docs/guide/embedding.md)
  * [文生图Image](./docs/guide/image.md)
* API文档
  * [对话补全ChatCompletion](./docs/api_reference/chat_completion.md)
  * [语义向量Embedding](./docs/api_reference/embedding.md)
  * [文生图Image](./docs/api_reference/image.md)


## Acknowledgement

我们借鉴了[OpenAI Python Library](https://github.com/openai/openai-python)部分API设计，在此对OpenAI Python Library作者及其开源社区表示感谢。

## License

ERNIE Bot SDK遵循Apache-2.0开源协议。
