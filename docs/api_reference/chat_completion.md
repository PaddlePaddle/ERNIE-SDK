# ChatCompletion

给定对话文本，模型服务会响应给出新的回复，包括回复的文本以及Token统计等信息。

## 函数接口

``` {.py .copy}
erniebot.ChatCompletion.create(**kwargs: Any)
	-> Union[EBResponse, Iterator[EBResponse]]
```

## 输入参数

| 参数名 | 类型 | 必填 | 描述 |
| :---   | :--- | :------- | :---- |
| model  | string | 是 | 模型名，当前支持"ernie-bot-3.5"和"ernie-bot-turbo" |
| messages | list(dict) | 是 | 对话上下文信息，其中list元素个数须为奇数 |
| top_p | float | 否 | 生成环节在概率加和为top_p以内的Top Token集合内进行采样 <br>(1)影响输出文本的多样性，取值越大，生成文本的多样性越强 <br>(2)默认0.8，取值范围 [0, 1.0] <br>(3)建议该参数和temperature只设置1个|
| temperature | float | 否 | 采样环节的参数，用于控制随机性 <br>(1)较高的数值会使输出更加随机，而较低的数值会使其更加集中和确定 <br>(2)默认0.95，范围 (0, 1.0]，不能为0 <br>(3)建议该参数和top_p只设置1个  |
| penalty_score | float | 否 | 通过对已生成的token增加惩罚，减少重复生成的现象, 值越高则惩罚越大 <br>(1)值越大表示惩罚越大 <br>(2)默认1.0，取值范围：[1.0, 2.0]|
| stream | boolean | 否 | 是否以流式接口返回数据，默认False |
| user_id | string | 否 | 表示最终用户的唯一标识符，可以监视和检测滥用行为，防止接口恶意调用 |

其中message作为python列表，其每个元素均为一个dict，包含"role"和"content"两个key，示例如下所示, 为了与模型进行多轮对话，我们将模型的回复结果插入在message中再继续请求
```
[
    {
        "role": "user",
        "content": "你好啊"
    },
    {
        "role": "assistant",
        "content": "你好，我是文心一言"
    },
    {
        "role": "user",
        "content": "深圳周末去哪里玩好?"
    }
]
```
| 键值 | 类型 | 描述 |
|:--- | :---- | :---- |
| role | string | user表示用户，assistant表示对话助手 |
| content | string | 对话内容，不能为空 |


## 返回结果

当为非流式，即`stream`为`False`时，接口返回`erniebot.response.EBResponse`结构体；当为流式，即`stream`为`True`时，接口返回Python的`Generator`类型结构体，其中`Generator`中每个元素均为`erniebot.response.EBResponse`结构体。

`erniebot.response.EBResponse`结构体示例数据如下所示：
```
{
    "code": 200,
    "id": "as-0rphgw7hw2",
    "object": "chat.completion",
    "created": 1692875360,
    "result": "深圳有很多不同的地方可以周末去玩，以下是一些推荐：\n\n1. 深圳东部：深圳东部有着美丽的海滩和壮观的山脉，是进行户外活动和探险的好地方。你可以去大梅沙海滨公园、小梅沙海洋世界、南澳岛等地方。\n2. 深圳中心城区：这里有许多购物中心、美食街、夜市等，可以品尝各种美食，逛街购物。你也可以去世界之窗、深圳华侨城等主题公园。\n3. 深圳西部：深圳西部有许多历史文化名胜和自然风光，比如深圳大学城、蛇口海上世界、南山海岸城等。\n4. 深圳郊区：深圳郊区有许多农业观光园、水果采摘园等，可以体验农家乐和亲近大自然。你可以去光明农场、欢乐田园等地方。\n5. 深圳室内：如果你想在周末找一个室内活动，可以去深圳的博物馆、艺术馆、电影院等，欣赏文化展览或者观看电影。\n\n以上是一些深圳周末游的推荐，你可以根据自己的兴趣和时间来选择合适的地方。",
    "is_truncated": false,
    "need_clear_history": false,
    "sentence_id": 0,
    "is_end": false,
    "usage": {
        "prompt_tokens": 8,
        "completion_tokens": 311,
        "total_tokens": 319
    }
}
```

其中字段含义如下表所示：

| 字段 | 类型 | 描述 |
| :--- | :---- | :---- |
| code | int | 请求返回状态 |
| body | dict | 请求返回的源数据 |
| result | string | 对话返回的生成结果 |
| is_truncated | boolean | 生成结果是否被长度限制截断 |
| sentence_id | int | 仅流式情况下返回该字段，表示返回结果中的文本顺序，从0开始计数 |
| need_clear_history | boolean | 表示用户输入是否存在安全，是否关闭当前会话，清理历史会话信息 <br>true：是，表示用户输入存在安全风险，建议关闭当前会话，清理历史会话信息 <br>false：否，表示用户输入无安全风险|
| ban_round | int | 当need_clear_history为true时，会返回此字段表示第几轮对话有敏感信息，如果是当前问题，ban_round=-1 |
| is_end | boolean | 仅流式情况下返回该字段，表示是否为是返回结果的最后一段文本 |
| usage | dict | 输入输出Token统计信息，注意当前Token统计采用估算逻辑， token数 = 汉字数 + 单词数 * 1.3。<br>prompt_tokens(int): 输入Token数量(含上下文拼接); <br>completion_tokens: 当前结构体包含的生成结果的Token数量; <br>total_tokens: 输入与输出的Token总数 |

## 使用示例

``` {.py .copy}
import erniebot

# erniebot.ak = "<EB-ACCESS-KEY-ID>"
# erniebot.sk = "<EB-SECRET-ACCESS-KEY>"

stream = False
chat_completion = erniebot.ChatCompletion.create(
    model="ernie-bot-3.5",
    messages=[{
        "role": "user",
        "content": "周末深圳去哪里玩？"
    }],
    top_p=0.95,
    stream=stream)

result = ""
if stream:
    for res in chat_completion:
        result += res.result
else:
    result = chat_completion.result

print("ERNIEBOT: ", result)
```
