# 对话补全（Chat Completion）

## 介绍

文心一言系列对话模型可以理解自然语言，并以文本输出与用户进行对话。将对话上下文与输入文本提供给模型，由模型给出新的回复，即为对话补全。

> 前往[yiyan.baidu.com](https://yiyan.baidu.com)直接体验文心一言对话能力。

对话补全功能可应用于广泛的实际场景，例如对话沟通、内容创作、分析控制等。

### 场景一：对话沟通

**匹配场景**：智能营销、智能客服、情感沟通等需要沟通对话的场景。

* 针对用户需求提供快速应答，精准匹配用户需求；
* 即时提供营销商拓、正向心理辅导等内容，提升用户体验。

**具体案例**：针对用户需求，输出推荐内容。例如，用户需要平台帮忙推荐一下四大名著，如下图所示：

![pic1](https://bce.bdstatic.com/doc/ai-cloud-share/WENXINWORKSHOP/image_a90f36c.png)

### 场景二：内容创作

**匹配场景**：剧本、故事、诗歌等文本创作场景。

* 根据用户的需求，生成精准匹配的创作文本，为用户提供视频编排的剧本来源；
* 润色成型的故事、诗歌等文本内容，给用户创造提升文本能力的文化环境。

**具体案例**：用户下发自定义指令，创作成型的文本内容。例如，用户需要平台按要求写一首藏头诗，如下图所示：

![pic2](https://bce.bdstatic.com/doc/ai-cloud-share/WENXINWORKSHOP/image_766ad39.png)

### 场景三：分析控制

**匹配场景**：代码生成、数据报表生成、内容分析等专业场景。

* 根据用户需求，快速生成可执行的代码；
* 根据用户需求，结合已有的多种数据，生成匹配度更高的应答内容。

**具体案例**：用户临时遇到需处理的问题，平台生成解决方案。例如，开发工程师利用平台生成具体代码，完成对代码的优化，如下图所示：

![pic3](https://bce.bdstatic.com/doc/ai-cloud-share/WENXINWORKSHOP/image_edb718d.png)

目前文心提供如下几种对话模型：

| 模型 | 说明 | API调用方式 |
| :--- | :--- | :----- |
| ernie-bot | 具备优秀的知识增强和内容生成能力，在文本创作、问答、推理和代码生成等方面表现出色。 |`erniebot.ChatCompletion.create(model='ernie-bot', ...)` |
| ernie-bot-turbo | 具备更快的响应速度和学习能力，API调用成本更低。 | `erniebot.ChatCompletion.create(model='ernie-bot-turbo', ...)` |
| ernie-bot-4 | 基于文心大模型4.0版本的文心一言，具备目前文心一言系列模型中最优的理解和生成能力。 | `erniebot.ChatCompletion.create(model='ernie-bot-turbo', ...)` |
| ernie-bot-8k | 在ernie-bot模型的基础上增强了对长对话上下文的支持，输入token数量上限为7000。 | `erniebot.ChatCompletion.create(model='ernie-bot-turbo', ...)` |

参阅[ChatCompletion API文档](../api_reference/chat_completion.md)了解API的完整使用方式。

## 常见问题

### 为什么模型输出不一致？

模型通常会引入一定的随机性来确保生成结果的多样性，因此，即使在输入相同的情况下，模型每次的输出结果也可能发生变化。可以通过设置`top_p`和`temperature`参数来调节生成结果的随机性，但需要注意的是，随机性始终存在，用户不应该期望从模型处获得完全确定的生成结果。

### 模型的输入长度有限制吗？

文心一言模型对输入的token数量有限制。对于ernie-bot、ernie-bot-turbo和ernie-bot-4模型，输入的token数量不能超过3000；对于ernie-bot-8k模型，输入token数量的限制是7000。以下分别讨论单轮和多轮对话的情形：

* 单轮对话时，输入的token数量不能超出限制。
* 多轮对话时，最后一条消息的token数量不能超出限制。此外，如果最后一条消息的token数量没有超出限制，而对话上下文（包括历史消息）的token总量超过了限制，则模型会在拼接输入时遗忘较早的历史信息，只保留满足token数限制的最近的对话上下文作为输入。

### 如何计算token数量？

目前千帆和AI Studio平台均采用`汉字数 + 单词数 * 1.3`估算token总数。可以通过如下代码估计token数量：

```{.py .copy}
import erniebot.utils
num_tokens = erniebot.utils.token_helper.approx_num_tokens("你好，我是文心一言。")
```
