
# 自定义 Tool

此处的自定义 Tool 指本地 Tool，此处就拿 `CurrentTimeTool` 为示例来讲解。

## Tool 剖析

一个 Tool 如果想要 Agent 能够正常执行，需要配置：
* 输入和输出参数：让 Agent 能够从历史对话信息中抽取出对应的字段信息。
* Tool 的描述信息：让 Agent 知道什么时候该调用这个 Tool。
* Tool 实现的具体逻辑：具体业务逻辑。
* Tool 的对话示例：为了让 Agent 能够更精确的调用该 Tool。

以 `CurrentTimeTool` 为例，我们来看看如何实现一个 Tool，先看看整体代码：

```python
class CurrentTimeToolOutputView(ToolParameterView):
    current_time: str = Field(description="当前时间")


class CurrentTimeTool(Tool):
    description: str = "CurrentTimeTool 用于获取当前时间"
    ouptut_type: Type[ToolParameterView] = CurrentTimeToolOutputView

    async def __call__(self) -> Dict[str, str]:
        return {"current_time": datetime.strftime(datetime.now(), "%Y年%m月%d号 %点:%分:%秒")}

    @property
    def examples(self) -> List[Message]:
        return [
            HumanMessage("现在几点钟了"),
            AIMessage(
                "",
                function_call={
                    "name": self.tool_name,
                    "thoughts": f"用户想知道现在几点了，我可以使用{self.tool_name}来获取当前时间，并从其中获得当前小时时间。",
                    "arguments": "{}",
                },
                token_usage={"prompt_tokens": 5, "completion_tokens": 7},  # For test only
            ),
            HumanMessage("现在是什么时候？"),
            AIMessage(
                "",
                function_call={
                    "name": self.tool_name,
                    "thoughts": f"用户想知道现在几点了，我可以使用{self.tool_name}来获取当前时间",
                    "arguments": "{}",
                },
                token_usage={"prompt_tokens": 5, "completion_tokens": 7},  # For test only
            ),
        ]
```

### 输入和输出参数

```python
class CurrentTimeToolOutputView(ToolParameterView):
    current_time: str = Field(description="当前时间")

class CurrentTimeTool(Tool):
    ouptut_type: Type[ToolParameterView] = CurrentTimeToolOutputView
```

在此示例当中没有输入参数，只有输出参数。

输入和输出参数必须继承 `ToolParameterView` 基类，并且通过 `Field` 装饰器来指定参数的描述信息。Agent 会在调用 Tool 之前根据此处配置的描述信息抽取处对应的字段信息，然后输入给 Tool 执行。

!!! warn 注意
    每个 Tool 的返回结果都必须是字典类型数据。

### Tool 的描述信息

```python
class CurrentTimeTool(Tool):
    description: str = "CurrentTimeTool 用于获取当前时间"
```

实现的方式就是给派生 Tool：`CurrentTimeTool` 添加一个 `description` 属性，该属性值将会作为 Tool 的功能描述信息让 Agent 判断何时该调用 Tool。

此描述信息至关重要，决定着此 Tool 能够被正常调用，所以建议认真编写此字段信息。

## Tool 的具体逻辑

这里就是发挥想象的地方，有了对应想要的输入信息，可以编写具体的业务逻辑，实现 Tool 的功能。

比如：“今天下午我有课要上吗？” 这个功能中，“今天下午”就作为一个输入参数，通过调用学校课程表的 API 即可获取到当对应时间段内的课程表信息，然后将信息返回给 Agent，例如：

```python

class CourseTool(Tool):

    async def __call__(self, time: datetime):
        courses = await search_course(time) # call api
        return {"courses": courses}
```

Agent 会根据输出参数来润色对应的结果，比如：
* 当有课的时候，Agent 可能会润色成：查询到今天下午有一节：`高等数学`，时间是从下午两点到四点，地点在 `教室 102`，请注意提前进教室。
* 当没有课的时候，Agent 可能会润色成：今天没有课，你可以去做一些自己喜欢的事情了。

## Tool 的对话示例

当 Tool 数量过多时，此时 Agent 只根据 description 来判断是否要调用 Tool 是不够的，所以需要编写对话示例来让 Agent 更精准的调用 Tool。

只需要给派生 Tool 写一个 examples 属性即可：

```python
class CurrentTimeTool(Tool):
    @property
    def examples(self) -> List[Message]:
        return [
            HumanMessage("现在几点钟了"),
            AIMessage(
                "",
                function_call={
                    "name": self.tool_name,
                    "thoughts": f"用户想知道现在几点了，我可以使用{self.tool_name}来获取当前时间，并从其中获得当前小时时间。",
                    "arguments": "{}",
                },
                token_usage={"prompt_tokens": 5, "completion_tokens": 7},
            ),
            AIMessage("现在是下午两点三十五分。"),
            HumanMessage("现在是什么时候？"),
            AIMessage(
                "",
                function_call={
                    "name": self.tool_name,
                    "thoughts": f"用户想知道现在几点了，我可以使用{self.tool_name}来获取当前时间",
                    "arguments": "{}",
                },
                token_usage={"prompt_tokens": 5, "completion_tokens": 7},
            ),
            AIMessage("现在是2023年1月2日，今天是周四，下午两点三十五分。"),
        ]
```

通过 python 代码形式配置 `examples` 属性，数据格式是：`List[Message]`，且通常是：[HumanMessage, AIMessage(function_call), AIMessage, ..., HumanMessage, AIMessage(function_call), AIMessage, ...]
