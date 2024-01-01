# 语义向量（Embedding）

## 介绍

语义向量功能是指使用模型将输入文本转化为用数值表示的向量形式。这些向量后续可用于文本检索、信息推荐、知识挖掘等场景。

目前文心提供如下语义向量模型：

| 模型 | 说明 | API调用方式 |
| :--- | :--- | :--- |
| ernie-text-embedding | 支持计算最多384个token的文本的向量表示。 | `erniebot.Embedding.create(model="ernie-text-embedding", ...)` |

参阅[Embedding API文档](../api_reference/embedding.md)了解API的完整使用方式。

## 常见问题

### 如何使用模型生成的向量？

一个典型的应用是通过计算向量间的余弦相似度估计文本的相似程度，以此支持检索、聚类、推荐等文本任务。

### 对于超长文本如何处理？

对于超长文本，可以采用切片方式对文本进行预处理。具体而言：将原始文本切分为多个小段，每个小段满足token数量的限制；然后，分别计算每小段文本的向量，并根据任务灵活使用。例如，在文本相似度计算中，可以计算输入query与每小段文本的余弦相似度，取最大值作为输入query与原始文本的相似度。

### 如何计算token数量？

可以采用`汉字数 + 单词数 * 1.3`估算token总数。ERNIE Bot提供了估计token数量的函数：

```{.py .copy}
import erniebot.utils
num_tokens = erniebot.utils.token_helper.approx_num_tokens("你好，我是文心一言。")
```
