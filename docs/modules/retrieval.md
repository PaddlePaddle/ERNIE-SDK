# Retrieval 模块介绍
## 1. Retrieval 简介

Retrieval组件是在给定非结构化查询的情况下返回文档的接口。Retrieval组件不需要存储文档，只需要返回（或检索）文档，并支持对上游向量数据库数据的增删查改操作。是目前最流行的RAG技术最重要的组件之一。

retrieval也是Agents应用的一个重要基础组件，提供了一个检索接口，用户可以挂载自己的私有文档，作为工具为agent提供外部知识。retrieval支持自研的文心百中搜索外，还兼容[LangChain](https://python.langchain.com/docs/modules/data_connection/)，[LlamaIndex](https://docs.llamaindex.ai/en/stable/getting_started/starter_example.html)的retrieval组件，以及众多检索增强的策略。

## 2.Retrieval 组件

| retrieval 组件名称 | 功能描述 | 代码链接
| :--: | :--: | :--: |
| BaizhongSearch| 支持百度自研的文心百中搜索| [baizhong_search.py](../package/erniebot_agent/retrieval.md#erniebot_agent.retrieval.BaizhongSearch) |
