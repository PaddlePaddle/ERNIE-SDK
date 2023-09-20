# 对话补全（Chat Completion）

## 介绍

文心提供的生成模型ERNIE Bot可以理解自然语言，并以文本输出与用户进行对话。

> 前往[yiyan.baidu.com](https://yiyan.baidu.com)直接体验文心一言对话能力。

使用ERNIE Bot可以实现不同的功能，涵盖对话沟通、内容创作、分析控制等通用应用场景。

### 场景一：对话沟通

**匹配场景**：智能营销、智能客服、情感沟通等需要沟通对话的场景。

在实际生活中，针对用户需求提供快速应答，精准匹配用户需求，完成营销商拓、及时响应、正向心理辅导等内容，提升客户体验。

**具体案例**：针对用户需求，输出结果。例如，用户需要平台帮忙推荐一下四大名著，如下图所示：

![pic1](https://bce.bdstatic.com/doc/ai-cloud-share/WENXINWORKSHOP/image_a90f36c.png)

### 场景二：内容创作

**匹配场景**：剧本、故事、诗歌等文本创作场景。

根据用户的需求，生成精准匹配的创作文本，为用户提供视频编排的剧本来源；润色成型的故事、诗歌等文本内容，给用户创造提升文本能力的文化环境。

**具体案例**：用户下发自定义指令，创作成型的文本内容。例如，用户需要平台按要求写一首藏头诗，如下图所示：

![pic2](https://bce.bdstatic.com/doc/ai-cloud-share/WENXINWORKSHOP/image_766ad39.png)

### 场景三：分析控制

**匹配场景**：代码生成、数据报表、内容分析等深度学习的文本场景。

根据用户的需求快速生成可执行的代码或者根据用户的需求，平台结合自身已具备的多种数据，生成匹配度更高的应答内容。

**具体案例**：用户临时遇到需处理的问题，平台生成解决方案。例如，开发工程师利用平台生成具体代码，完成代码的优化，如下图所示：

![pic3](https://bce.bdstatic.com/doc/ai-cloud-share/WENXINWORKSHOP/image_edb718d.png)

目前文心提供如下两种生成式对话模型，

| 模型 | 说明 | API调用方式 |
| :--- | :--- | :----- |
| ernie-bot | 具备最优的知识增强和生成能力，在文本创作、问答、推理和代码生成等方面表现出色。 |`erniebot.ChatCompletion.create(model="ernie-bot", ...)` |
| ernie-bot-turbo | 具备更快的响应速度和学习能力，API调用成本更低。 | `erniebot.ChatCompletion.create(model="ernie-bot-turbo", ...)`|

参阅[ChatCompletion API文档](../api_reference/chat_completion.md)了解API的完整使用方式。

## 常见问题

### 为什么模型输出不一致？

生成模型在默认情况下会具有有一定的随机性来确保生成结果的多样性，因此即使同样的输入情况下每次的输出结果也会发生变化。可以通过设置top_p和temperrature参数来降低随机性，但并不能去除随机性。

### 模型的输入输出有什么限制？

ernie-bot与ernie-bot-turbo模型对于输入和输出的token数量会有限制，通常情况下输入的token数量不能超过3072，输出的token数量不会超过1024。当输入的token数量超过限制时，会有以下几种情况：

* 单轮对话下，用户发送的文本超出输入限制，会直接返回错误；
* 多轮对话下，用户发送的文本，如若最近一次用户文本超出输入限制，会直接返回错误；如若最近一次用户文本没有超出限制，则模型服务会在拼接历史信息时最多拼接到相应的token数上限，并丢弃多余的历史信息。

### 如何计算Token数量

目前千帆平台采用`汉字数 + 单词数 * 1.3`估算token总数。使用ERNIE Bot SDK，你可以通过如下代码计算得到token数量：

```{.py .copy}
import erniebot
token_num = erniebot.utils.approx_num_tokens("你好，我是文心一言。")
```
