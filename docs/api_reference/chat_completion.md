# ChatCompletion

给定对话文本，模型服务会响应给出新的回复，包括回复的文本以及token数量统计等信息。

## Python接口

```{.py .copy}
erniebot.ChatCompletion.create(**kwargs: Any)
	-> Union[EBResponse, Iterator[EBResponse]]
```

## 输入参数

| 参数名 | 类型 | 必填 | 描述 |
| :---   | :--- | :------- | :---- |
| model  | string | 是 | 模型名称。当前支持`'ernie-bot'`和`'ernie-bot-turbo'`。 |
| messages | list[dict] | 是 | 对话上下文信息。列表中的元素个数须为奇数。详见[`messages`](#messages)。 |
| functions | list[dict] | 否 | 可触发函数的描述列表。详见[`functions`](#functions)。`ernie-bot-turbo`模型暂不支持此参数。 |
| top_p | float | 否 | 生成环节在概率加和为`top_p`以内的top token集合内进行采样： <br>(1) 影响输出文本的多样性，取值越大，生成文本的多样性越强； <br>(2) 默认`0.8`，取值范围为`[0, 1.0]`； <br>(3) 建议该参数和temperature只设置其中一个。 |
| temperature | float | 否 | 采样环节的参数，用于控制随机性。 <br>(1) 较高的数值会使输出更加随机，而较低的数值会使其更加集中和确定； <br>(2) 默认`0.95`，范围为`(0, 1.0]`，不能为`0`； <br>(3) 建议该参数和`top_p`只设置其中一个。 |
| penalty_score | float | 否 | 通过对已生成的token增加惩罚，减少重复生成的现象，值越高则惩罚越大。 <br>(1) 值越大表示惩罚越大； <br>(2) 默认`1.0`，取值范围：`[1.0, 2.0]`。 |
| stream | boolean | 否 | 是否流式返回数据，默认`False`。 |
| user_id | string | 否 | 表示最终用户的唯一标识符，可以监视和检测滥用行为，防止接口恶意调用。 |

<details>
<summary><code name="messages">messages</code></summary>

`messages`为一个Python list，其中每个元素为一个dict。在如下示例中，为了与模型进行多轮对话，我们将模型的回复结果插入在`messages`中再继续请求：

```{.py .copy}
[
    {
        'role': 'user',
        'content': "你好啊"
    },
    {
        'role': 'assistant',
        'content': "你好，我是文心一言"
    },
    {
        'role': 'user',
        'content': "深圳周末去哪里玩好?"
    }
]
```

`messages`中的每个元素包含如下键值对：

| 键名 | 值类型 | 必填 | 值描述 |
|:--- | :---- | :--- | :---- |
| role | string | 是 | `'user'`表示用户，`'assistant'`表示对话助手，`'function'`表示函数。 |
| content | string or `None` | 是 | 当`role`不为`'function'`时，表示对话内容，必须设置该参数为非`None`值；当`role`为`'function'`时，表示函数响应参数，可以设置该参数为`None`。 |
| name | string | 否 | 信息的作者。当`role='function'`时，此参数必填，且是`function_call`中的`name`。 |
| function_call | dict | 否 | 由模型生成的函数调用，包含函数名称和请求参数等。 |

`function_call`为一个Python dict，其中包含如下键值对：

| 键名 | 值类型 | 必填 | 值描述 |
|:--- | :---- | :--- | :---- |
| name | string | 是 | 函数名称。 |
| thoughts | string | 否 | 模型思考过程。 |
| arguments | string | 是 | 请求参数。 |

</details>

<details>
<summary><code name="functions">functions</code></summary>

`functions`为一个Python list，其中每个元素为一个dict。示例如下：

```{.py .copy}
[
    {
        'name': 'get_current_temperature',
        'description': "获取指定城市的气温",
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': "城市名称"
                },
                'unit': {
                    'type': 'string',
                    'enum': [
                        '摄氏度',
                        '华氏度'
                    ]
                }
            },
            'required': [
                'location',
                'unit'
            ]
        },
        'responses': {
            'type': 'object',
            'properties': {
                'temperature': {
                    'type': 'integer',
                    'description': "城市气温"
                },
                'unit': {
                    'type': 'string',
                    'enum': [
                        '摄氏度',
                        '华氏度'
                    ]
                }
            }
        }
    }
]
```

`functions`中的每个元素包含如下键值对：

| 键名 | 值类型 | 必填 | 值描述 |
|:--- | :---- | :--- | :---- |
| name | string | 是 | 函数名称。 |
| description | string | 是 | 对函数功能的描述。 |
| parameters | dict | 是 | 函数请求参数。采用[JSON Schema](https://json-schema.org/)格式。 |
| responses | dict | 否 | 函数响应参数。采用[JSON Schema](https://json-schema.org/)格式。 |
| examples | list[dict] | 否 | 函数调用示例。可提供与`messages`类似的对话上下文信息作为函数调用的例子。一个例子如下：`[{'role': 'user', 'content': "深圳市今天气温如何？"}, {'role': 'assistant', 'content': None, 'function_call': {'name': 'get_current_temperature', 'arguments': '{"location":"深圳市","unit":"摄氏度"}'}}, {'role': 'function', 'name': 'get_current_temperature', 'content': '{"temperature":25,"unit":"摄氏度"}'}]` |
| plugin_id | string | 否 | 标记函数关联的插件，便于数据统计。 |

</details>

## 返回结果

当采用非流式模式（即`stream`为`False`）时，接口返回`erniebot.response.EBResponse`对象；当采用流式模式（即`stream`为`True`）时，接口返回一个Python生成器，其产生的每个元素均为`erniebot.response.EBResponse`对象。

`erniebot.response.EBResponse`对象中包含一些字段。一个典型示例如下：

```python
{
    'rcode': 200,
    'id': 'as-0rphgw7hw2',
    'object': 'chat.completion',
    'created': 1692875360,
    'result': "深圳有很多不同的地方可以周末去玩，以下是一些推荐：\n\n1. 深圳东部：深圳东部有着美丽的海滩和壮观的山脉，是进行户外活动和探险的好地方。你可以去大梅沙海滨公园、小梅沙海洋世界、南澳岛等地方。\n2. 深圳中心城区：这里有许多购物中心、美食街、夜市等，可以品尝各种美食，逛街购物。你也可以去世界之窗、深圳华侨城等主题公园。\n3. 深圳西部：深圳西部有许多历史文化名胜和自然风光，比如深圳大学城、蛇口海上世界、南山海岸城等。\n4. 深圳郊区：深圳郊区有许多农业观光园、水果采摘园等，可以体验农家乐和亲近大自然。你可以去光明农场、欢乐田园等地方。\n5. 深圳室内：如果你想在周末找一个室内活动，可以去深圳的博物馆、艺术馆、电影院等，欣赏文化展览或者观看电影。\n\n以上是一些深圳周末游的推荐，你可以根据自己的兴趣和时间来选择合适的地方。",
    'is_truncated': false,
    'need_clear_history': false,
    'sentence_id': 0,
    'is_end': false,
    'usage': {
        'prompt_tokens': 8,
        'completion_tokens': 311,
        'total_tokens': 319
    }
}
```

`erniebot.response.EBResponse`对象的各关键字段含义如下表所示：

| 字段名 | 类型 | 描述 |
| :--- | :---- | :---- |
| rcode | int | HTTP响应状态码。 |
| result | string | 对话返回的生成结果。 |
| is_truncated | boolean | 生成结果是否被长度限制截断。 |
| sentence_id | int | 仅流式模式下返回该字段，表示返回结果中的文本顺序，从`0`开始计数。 |
| need_clear_history | boolean | 表示用户输入是否存在安全，是否关闭当前会话，清理历史会话信息 <br>`True`：是，表示用户输入存在安全风险，建议关闭当前会话，清理历史会话信息 <br>`False`：否，表示用户输入无安全风险。 |
| ban_round | int | 当`need_clear_history`为`True`时，会返回此字段表示第几轮对话有敏感信息，如果是当前轮次存在问题，则`ban_round=-1`。 |
| is_end | boolean | 仅流式模式下返回该字段，表示是否为是返回结果的最后一段文本。 |
| usage | dict | 输入输出token统计信息。token数量采用如下公式估算：`token数 = 汉字数 + 单词数 * 1.3`。<br>`prompt_tokens`：输入token数量（含上下文拼接）；<br>`completion_tokens`：当前生成结果包含的token数量；<br>`total_tokens`：输入与输出的token总数；<br> `plugins`：插件消耗的token数量。 |
| function_call | dict | 由模型生成的函数调用，包含函数名称和请求参数等。详见[`messages`](#messages)中的`function_call`。 |

假设`resp`为一个`erniebot.response.EBResponse`对象，字段的访问方式有2种：`resp['result']`或`resp.result`均可获取`result`字段的内容。此外，可以使用`resp.get_result()`获取响应中的“主要结果”：当模型返回函数调用信息时（此时，`resp`具有`function_call`字段），`resp.get_result()`的返回结果与`resp.function_call`一致；否则，`resp.get_result()`的返回结果与`resp.result`一致，即模型给出的回复文本。

## 使用示例

```{.py .copy}
import erniebot

erniebot.api_type = 'aistudio'
erniebot.access_token = '<access-token-for-aistudio>'

stream = False
response = erniebot.ChatCompletion.create(
    model='ernie-bot',
    messages=[{
        'role': 'user',
        'content': "周末深圳去哪里玩？"
    }],
    top_p=0.95,
    stream=stream)

result = ""
if stream:
    for resp in response:
        result += resp.get_result()
else:
    result = response.get_result()

print("ERNIEBOT: ", result)
```
