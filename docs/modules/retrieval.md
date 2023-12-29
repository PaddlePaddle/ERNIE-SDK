# Retrieval 模块介绍

## 1. Retrieval 简介

检索（Retrieval）就是从海量的信息中查找与查询相关的内容。现在检索按照技术分类可以分为关键字检索和语义检索，关键字检索很常见，比如现在的搜索引擎，学术文献搜索大多都基于关键字的全文检索及其一些优化技术；而义检索，是指检索系统不再拘泥于用户查询（Query）字面本身，而是能精准捕捉到用户查询后面的真正意图并以此来搜索，从而更准确地向用户返回最符合的结果。通过使用最先进的语义索引模型找到文本的向量表示，在高维向量空间中对它们进行索引，并度量查询向量与索引文档的相似程度，从而解决了关键词索引带来的缺陷。

例如下面两组文本 Pair，如果基于关键词去计算相似度，两组的相似度是相同的。而从实际语义上看，第一组相似度高于第二组。

```
车头如何放置车牌    前牌照怎么装
车头如何放置车牌    后牌照怎么装
```

Retrieval组件是一种接口，它能够在接收到非结构化查询时返回相关文档。无论是通过关键词检索还是语义检索，该组件都能有效地满足查询需求。值得注意的是，Retrieval组件无需存储文档，其主要功能在于返回或检索文档，并且能够灵活地对上游数据库进行数据的增加、删除、查找和修改操作。因此，它被广泛认为是当前最流行的RAG技术中不可或缺的重要组成部分。

另外，Retrieval组件也是Agents应用的一个重要基础组件，提供了一个检索接口，用户可以挂载自己的私有文档，作为工具为agent提供外部知识。retrieval支持自研的文心百中搜索外，还兼容[LangChain](https://python.langchain.com/docs/modules/data_connection/)，[LlamaIndex](https://docs.llamaindex.ai/en/stable/getting_started/starter_example.html)的retrieval组件，以及众多检索增强的策略。

## 2.Retrieval 组件

| retrieval 组件名称 | 功能描述 | 代码链接
| :--: | :--: | :--: |
| BaizhongSearch| 支持百度自研的文心百中搜索| [baizhong_search.py](../package/erniebot_agent/retrieval.md#erniebot_agent.retrieval.BaizhongSearch) |
