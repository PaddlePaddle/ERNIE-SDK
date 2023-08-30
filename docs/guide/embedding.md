# Embedding

## 介绍

Embedding服务是文心基于大模型技术研发的文本表示模型，将文本转化为用数值表示的向量形式，用于文本检索、信息推荐、知识挖掘等场景。

目前文心提供如下向量表示模型：

| 模型 | 说明 | API接口 |
| :--- | :---- | :----- |
| ernie-text-embedding | 支持计算最多384个Token长度的文本的向量表示 | `erniebot.Embedding.create(model="ernie-text-embedding", ...)` |

参阅[Embedding API](../api_reference/embedding.md)了解完整的API文档和使用方式。

## 常见问题

### 怎么使用计算得到的向量？

我们可以对两段文本的向量，通过计算向量间的余弦相似度得到其相似度打分，以此支持检索、聚类、推荐等文本任务。

### 对于超长文本如何处理？

超长文本可以采用切片方式将原始文本切分为多个小段的文本，满足Token长度的限制，分别计算每个小段文本的向量，并根据任务灵活使用。例如在文本相似度计算中，我们可以计算输入query与每小段文本的余弦相似度，取最大值作为输入query与原始文本的相似度。


### 如何计算Token数量
目前千帆平台采用`汉字数 + 单词数 * 1.3`来估算得到Token总数，在ERNIE Bot SDK中，你可以通过如下代码计算得到Token数量
```
import erniebot
token_num = erniebot.utils.approx_num_tokens("你好，我是文心一言。")
```
