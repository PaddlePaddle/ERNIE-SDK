# Embedding

使用模型将文本转化为用数值表示的向量形式。

## Python接口

```{.py .copy}
erniebot.Embedding.create(
    model: str,
    input: List[str],
    *,
    user_id: Optional[str] = ...,
    headers: Optional[HeadersType] = ...,
    request_timeout: Optional[float] = ...,
    _config_: Optional[ConfigDictType] = ...,
) -> EmbeddingResponse
```

## 输入参数

| 参数名 | 类型 | 必填 | 描述 |
| :--- | :--- | :--- | :--- |
| model | str | 是 | 模型名称。当前支持`"ernie-text-embedding"`。 |
| input | list[str] | 是 | 输入的文本列表，列表中每个元素为一段单独的文本。注意：<ul><li>列表长度有最大限制。对于ernie-text-embedding模型，列表长度不能超过16。</li><li>每段文本的token数量有最大限制，超出限制则报错（可以采用<code>汉字数 + 单词数 * 1.3</code>估算token数量）。对于ernie-text-embedding模型，每段文本支持最多384个token。</li><li>文本内容不能为空。</li></ul> |
| user_id | str | 否 | 终端用户的唯一标识符，可以监视和检测滥用行为，防止接口被恶意调用。 |
| headers | dict | 否 | 附加的HTTP请求头。 |
| request_timeout | float | 否 | 单个HTTP请求的超时时间，单位为秒。 |
| \_config\_ | dict | 否 | 用于覆盖全局配置。 |

## 返回结果

接口返回`erniebot.EmbeddingResponse`对象。

返回结果的一个典型示例如下：

```{.py .no-copy}
{
    "rcode": 200,
    "id": "as-s0tdsgnuu4",
    "object": "embedding_list",
    "created": 1692933427,
    "data": [
        {
            "object": "embedding",
            "embedding": [
                0.12393086403608322,
                0.06512520462274551,
                0.05346716567873955,
                ...
            ],
            "index": 0
        },
        {
            "object": "embedding",
            "embedding": [
                0.12393086403608322,
                0.06512520462274551,
                0.05346716567873955,
                ...
            ],
            "index": 1
        }
    ],
    "usage": {
        "prompt_tokens": 98,
        "total_tokens": 98
    }
}
```

其中关键字段含义如下表所示：

| 字段名 | 类型 | 描述 |
| :--- | :--- | :--- |
| rcode | int | HTTP响应状态码。 |
| data | list[dict] | 向量列表，列表中元素个数与输入的文本数量一致。列表中的元素均为dict，包含如下键值对：<ul><li><code>object</code>：固定为<code>"embedding"</code>。</li><li><code>embedding</code>：模型生成的向量。对于ernie-text-embedding模型，向量维度为384。</li><li><code>index</code>：序号。</li></ul> |
| usage | dict | 输入、输出token统计信息。<ul><li><code>prompt_tokens</code>：输入token数量；</li><li><code>total_tokens</code>：输入与输出的token总数。</li></ul> |

假设`resp`为一个`erniebot.EmbeddingResponse`对象，字段的访问方式有2种：`resp["data"]`或`resp.data`均可获取`data`字段的内容。此外，可以使用`resp.get_result()`获取响应中的“主要结果”。具体而言，`resp.get_result()`返回一个Python list，其中顺序包含每段输入文本的向量结果。

## 使用示例

```{.py .copy}
import erniebot
import numpy as np

erniebot.api_type = "aistudio"
erniebot.access_token = "<access-token-for-aistudio>"

response = erniebot.Embedding.create(
    model="ernie-text-embedding",
    input=[
        "我是百度公司开发的人工智能语言模型，我的中文名是文心一言，英文名是ERNIE-Bot，可以协助您完成范围广泛的任务并提供有关各种主题的信息，比如回答问题，提供定义和解释及建议。如果您有任何问题，请随时向我提问。",
        "2018年深圳市各区GDP"
    ])

for embedding in response.get_result():
    embedding = np.array(embedding)
    print(embedding)
```
