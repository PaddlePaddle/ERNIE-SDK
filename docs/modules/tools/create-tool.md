
# 创建 Tool

在此，我们所提及的“自定义 Tool”指的是本地 Tool。为了更具体地阐述这一概念，我们将以 CurrentTimeTool 为例进行详细讲解。

## 1. Tool 剖析

为了确保一个工具（Tool）能够被代理（Agent）正常执行，需要进行以下配置：

1. 输入和输出参数：通过配置输入和输出参数，Agent能够从历史对话信息中提取相应的字段信息，并与工具进行有效的交互。
2. 工具的描述信息：这部分提供了关于工具用途和功能的详细说明。Agent根据这些描述信息来判断何时应该调用该工具。描述信息应包括工具的功能、适用场景以及任何特定的调用要求。
3. 工具的具体逻辑：这是工具的核心部分，包含了实现特定业务逻辑的代码或指令。具体逻辑定义了工具在接收到代理的请求时应如何响应和执行相应的任务。
4. 工具的对话示例：这些示例用于演示工具在实际对话中的用法。通过提供对话示例，代理可以更准确地理解如何调用该工具，并在适当的情境下应用它。对话示例还可以帮助开发者测试和验证工具的有效性和正确性。

以 CurrentTimeTool 为例，下面我们将展示如何实现一个工具。首先，让我们查看整体代码结构...

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

## 2. 步骤

### 2.1 输入和输出参数

```python
class CurrentTimeToolOutputView(ToolParameterView):
    current_time: str = Field(description="当前时间")

class CurrentTimeTool(Tool):
    ouptut_type: Type[ToolParameterView] = CurrentTimeToolOutputView
```

在此示例当中没有输入参数，只有输出参数。

输入和输出参数必须继承 `ToolParameterView` 基类，并且通过 `Field` 类来指定参数的详细信息，其中包含：

* 类型注释：`current_time: str` 中的 str 就是指定此字段的类型信息。
* 字段描述：`description`，主要用于从历史对话信息中提取出对应字段信息或者根据字典信息进行润色。
* 字段默认值：如果在历史对话当中没有提取对应字段信息，可初始化默认值。

Agent 在调用 Tool 之前，会通过 FunctionCall 抽取出对应字段信息，然后输入给 Tool 执行。

!!! warn 注意
    每个 Tool 的返回结果都必须是字典类型数据。

### 2.2 Tool 的描述信息

```python
class CurrentTimeTool(Tool):
    description: str = "CurrentTimeTool 用于获取当前时间"
```

实现的方式就是给派生 Tool：`CurrentTimeTool` 添加一个 `description` 属性，该属性值将会作为 Tool 的功能描述信息让 Agent 判断何时该调用 Tool。

此描述信息至关重要，决定着此 Tool 能够被正常调用，所以建议认真编写此字段信息。

### 2.2 Tool 的具体逻辑

这里就是发挥想象的地方，有了对应想要的输入信息，可以编写具体的业务逻辑，实现 Tool 的功能。

比如：“今天下午我有课要上吗？” 这个功能中，“今天下午”就作为一个输入参数，通过调用学校课程表的 API 即可获取到当对应时间段内的课程表信息，然后将信息返回给 Agent，例如：

```python

class CourseToolCourseOutputView(ToolParameterView):
    name: str = Field(description="课程名称")
    time: str = Field(description="上课时间段")
    room: str = Field(description="教室地点")

class CourseToolOutputView(ToolParameterView):
    courses: List[CourseToolCourseOutputView] = Field(description="课程信息")

class CourseTool(Tool):
    ouptut_type: Type[ToolParameterView] = CourseToolOutputView

    async def __call__(self, time: datetime):
        courses = await search_course(time) # call api
        return {"courses": courses}
```

Agent 会根据输出参数来润色对应的结果，比如：
* 当有课的时候，Agent 可能会润色成：查询到今天下午有一节：`高等数学`，时间是从下午两点到四点，地点在 `教室 102`，请注意提前进教室。
* 当没有课的时候，Agent 可能会润色成：今天没有课，你可以去做一些自己喜欢的事情了。

### 2.3 Tool 的对话示例

当 Tool 数量过多时，此时 Agent 只根据 description 来判断是否要调用具体 Tool 是不够的， 此时可通过编写更多的对话示例来让 Agent 更精准的调用 Tool。

实现方式是只需要给派生 Tool 写一个 examples 属性即可：

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
