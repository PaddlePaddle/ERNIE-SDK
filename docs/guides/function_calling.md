# 函数调用

## 介绍

文心一言提供函数调用功能，模型根据用户需求以及对函数的描述确定何时以及如何调用函数。具体而言，一个典型的函数调用流程如下：

1. 用户提供对一组函数的名称、功能、请求参数（输入参数）和响应参数（返回值）的描述，并以自然语言阐述需求；
2. 模型根据用户需求以及函数描述信息，智能确定是否应该调用函数、调用哪一个函数、以及在调用该函数时需要如何设置输入参数；
3. 用户根据模型的提示调用函数，并将函数的响应传递给模型；
4. 模型综合对话上下文信息，以自然语言形式给出满足用户需求的回答。

借由函数调用，用户可以从大模型获取结构化数据，进而利用编程手段将大模型与已有的内外部API结合以构建应用。

在ERNIE Bot SDK中，`erniebot.ChatCompletion.create`接口提供函数调用功能。关于该接口的更多详情请参考[ChatCompletion API文档](../api_reference/chat_completion.md)。

# 使用示例

如下展示了一个函数调用的例子。

首先对函数的基本信息进行描述，使用[JSON Schema](https://json-schema.org/)格式描述函数的请求参数与响应参数。

``` {.py .copy}
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
                    "type": "int",
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

接着，将以上信息与对需要完成的任务的自然语言描述一同传入`erniebot.ChatCompletion` API。

``` {.py .copy}
import erniebot

erniebot.api_type = "qianfan"
erniebot.ak = "<ak-for-qianfan>"
erniebot.sk = "<sk-for-qianfan>"

messages = [
    {
        "role": "user",
        "content": "深圳市今天气温如何？"
    }
]

response = erniebot.ChatCompletion.create(
    model="ernie-bot-3.5",
    messages=messages,
    functions=functions
)
assert hasattr(response, "function_call")
function_call = response.function_call
print(function_call)
```

上述代码中的断言语句用于确保`response`中包含`function_call`字段。在实际生产中通常还需要考虑`response`中不包含`function_call`的情况，这意味着模型选择不调用任何函数。上述代码输出结果可能如下（由于大模型生成结果具有不确定性，执行上述代码的结果与本示例不一定一致）：

```text
{'name': 'get_current_temperature', 'thoughts': '我需要获取指定城市的气温', 'arguments': '{"unit":"摄氏度","location":"深圳市"}'}
```

`function_call`是一个Python dict，其中包含的键`name`、`thoughts`分别对应大模型选择调用的函数名称以及模型的思考过程。`function_call["arguments"]`是一个JSON格式的字符串，其中包含了调用函数时需要用到的参数。

解析`function_call["arguments"]`，并使用解析得到的参数调用对应的函数。本示例使用一个硬编码的dummy函数作为演示，在实际生产中可将其替换为真正具备相应功能的API。

``` {.py .copy}
import json

def get_current_temperature(location: str, unit: str) -> dict:
    return {"temperature": 25, "unit": "摄氏度"}

name2function = {"get_current_temperature": get_current_temperature}
func = name2function[function_call["name"]]
args = json.loads(function_call["arguments"])
res = get_current_temperature(location=args["location"], unit=args["unit"])
```

将模型上一轮的响应以及函数的响应加入到对话上下文信息中，再次传递给模型。如果函数的响应不是JSON格式的字符串，需要先对其进行编码。

``` {.py .copy}
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
        "content": json.dumps(res)
    }
)
response = erniebot.ChatCompletion.create(
    model="ernie-bot-3.5",
    messages=messages,
    functions=functions
)
print(response.result)
```

输出结果可能如下：

```text
深圳市今天的温度是25摄氏度，天气还算舒适，建议穿轻薄的衣服出门。
```
