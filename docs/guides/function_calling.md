# 函数调用（Function Calling）

## 介绍

文心一言提供函数调用功能，模型根据用户需求以及对函数的描述确定何时以及如何调用函数。具体而言，一个典型的函数调用流程如下：

(1) 用户提供对一组函数的名称、功能、请求参数（输入参数）和响应参数（返回值）的描述；
<br>(2) 模型根据用户需求以及函数描述信息，智能确定是否应该调用函数、调用哪一个函数、以及在调用该函数时需要如何设置输入参数；
<br>(3) 用户根据模型的提示调用函数，并将函数的响应传递给模型；
<br>(4) 模型综合对话上下文信息，以自然语言形式给出满足用户需求的回答。

借由函数调用，用户可以从大模型获取结构化数据，进而利用编程手段将大模型与已有的内外部API结合以构建应用。

在ERNIE Bot SDK中，`erniebot.ChatCompletion.create`接口提供函数调用功能。关于该接口的更多详情请参考[ChatCompletion API文档](../api_reference/chat_completion.md)。

# 使用示例

假设我们有如下的函数实现：

```{.py .copy}
def get_current_temperature(location: str, unit: str) -> dict:
    return {"temperature": 25, "unit": "摄氏度"}
```

在接下来的例子中，我们将尝试让大模型“指导”我们调用这个函数以完成指定的任务。需要说明的是，`get_current_temperature`是出于演示目的定义的一个硬编码的dummy函数，在实际生产中可将其替换为真正具备相应功能的API。

(1) 首先，对函数的基本信息进行描述，使用[JSON Schema](https://json-schema.org/)格式描述函数的请求参数与响应参数。

```{.py .copy}
functions = [
    {
        "name": "get_current_temperature",
        "description": "获取指定城市的气温",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市名称"
                },
                "unit": {
                    "type": "string",
                    "enum": [
                        "摄氏度",
                        "华氏度"
                    ]
                }
            },
            "required": [
                "location",
                "unit"
            ]
        },
        "responses": {
            "type": "object",
            "properties": {
                "temperature": {
                    "type": "integer",
                    "description": "城市气温"
                },
                "unit": {
                    "type": "string",
                    "enum": [
                        "摄氏度",
                        "华氏度"
                    ]
                }
            }
        }
    }
]
```

代码中定义了一个列表`functions`，作为示例，其中仅包含对一个函数`get_current_temperature`的名称、请求参数等信息的描述。

(2) 接着，将以上信息与对需要完成的任务的自然语言描述一同传给`erniebot.ChatCompletion` API。

```{.py .copy}
import erniebot

erniebot.api_type = "aistudio"
erniebot.access_token = "<eb-access-token>"

messages = [
    {
        "role": "user",
        "content": "深圳市今天气温如何？"
    }
]

response = erniebot.ChatCompletion.create(
    model="ernie-bot",
    messages=messages,
    functions=functions
)
assert hasattr(response, "function_call")
function_call = response.function_call
print(function_call)
```

以上代码中的断言语句用于确保`response`中包含`function_call`字段。在实际生产中通常还需要考虑`response`中不包含`function_call`的情况，这意味着模型选择不调用任何函数。以上代码的输出结果可能如下（由于大模型生成结果具有不确定性，执行上述代码的结果与本示例不一定一致）：

```text
{'name': 'get_current_temperature', 'thoughts': '我需要获取指定城市的气温', 'arguments': '{"unit":"摄氏度","location":"深圳市"}'}
```

`function_call`是一个字典，其中包含的键`name`、`thoughts`分别对应大模型选择调用的函数名称以及模型的思考过程。`function_call["arguments"]`是一个JSON格式的字符串，其中包含了调用函数时需要用到的参数。

(3) 然后，根据模型的提示调用相应函数得到结果。

```{.py .copy}
import json

name2function = {"get_current_temperature": get_current_temperature}
func = name2function[function_call["name"]]
args = json.loads(function_call["arguments"])
res = func(location=args["location"], unit=args["unit"])
```

以上代码从`function_call`中获取模型选择调用的函数名称（`function_call["name"]`），通过该名称找到对应的函数，并从`function_call["arguments"]`中解析需要传入函数的参数，最终完成对函数的调用。

(4) 最后，将模型上一轮的响应以及函数的响应加入到对话上下文信息中，再次传递给模型。回传给模型的函数响应内容应当是JSON格式的字符串（如`'{"temperature": 25, "unit": "摄氏度"}'`），在本示例中，函数的响应是一个字典，因此需要先调用`json.dumps`函数对其进行编码。

```{.py .copy}
messages.append(
    {
        "role": "assistant",
        "content": None,
        "function_call": function_call
    }
)
messages.append(
    {
        "role": "function",
        "name": function_call["name"],
        "content": json.dumps(res, ensure_ascii=False)
    }
)
response = erniebot.ChatCompletion.create(
    model="ernie-bot",
    messages=messages,
    functions=functions
)
print(response.result)
```

输出结果可能如下：

```text
深圳市今天的温度是25摄氏度，天气还算舒适，建议穿轻薄的衣服出门。
```
