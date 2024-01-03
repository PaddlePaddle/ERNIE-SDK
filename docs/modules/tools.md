# Tool 模块

## 1. 模块介绍

Tool 模块是智能体（Agent）的辅助工具，可以用来执行一些特定的任务，如获取当前时间、执行 Python 代码以及调用内部某 API 执行特定的功能等，此模块极大扩展了智能体的能力边界，也是未来智能体（Agent）能够大规模接入到传统软件中的一个关键组成部分。

Tool 模块分为 LocalTool 和 RemoteTool 两类，LocalTool 运行在本地的机器上，RemoteTool 则运行在远端。

## 2. 核心类

下面简单介绍 `Tool` 模块的核心类，详细接口请参考 [API 文档](../package/erniebot_agent/tools.md)

### 2.1 Tool

此类为所有 LocalTool 的积累，为一个抽象类，需要子类继承并实现 `__call__` 方法，示例代码如下所示：

```python
class CurrentTimeTool(Tool):
    description: str = "CurrentTimeTool 用于获取当前时间"
    ouptut_type: Type[ToolParameterView] = CurrentTimeToolOutputView

    async def __call__(self) -> Dict[str, str]:
        return {"current_time": datetime.strftime(datetime.now(), "%Y年%m月%d日 %H时%M分%S秒")}
```

!!! notes 注意

    - `__call__` 为异步方法，必须要添加 `async` 关键字才能执行。

### 2.2 ToolParameteView

`ToolParameteView` 用于定义工具的输入和输出参数的元信息，一方面用于生成 `function_call` 的 `openapi`结构数据，另一方面可用于做模型抽取参数的校验工作，比如上述 `CurrentTimeToolOutputView` 的实际代码如下所示：

```python
class CurrentTimeToolOutputView(ToolParameterView):
    current_time: str = Field(description="当前时间")
```

这样就可以给工具的返回值定义一个 `current_time` 字段，并且可以给这个字段添加描述信息，最终会生成 `openapi` 结构数据让大模型来润色成自然语言。

这里展示了输出参数的定义，输入参数的定义类似。

### 2.3 RemoteToolkit

用于初始化远端的工具集，比如 [使用 RemoteTool](../quickstart/use-tool.md) 章节中介绍的使用 `RemoteToolkit` 的示例，可从 AI Studio 工具中心或某一个 url 中初始化工具集的元信息。

而 RemoteToolkit 需要的元信息是从远端的 `http://www.xxx.com/well-known/openapi.yaml` 文件中下载并获得的。

!!! Tips

    一个远端工具所有的描述和调用信息都可以通过标准的 [OpenAPI](https://swagger.io/specification/) 规范来定义，而 `RemoteToolkit` 的初始化就是基于此标准来。

    也就是说只要对应的远端工具集的 `openapi.yaml` 文件符合 OpenAPI 标准，就可以使用 RemoteToolkit 进行初始化，并导入到 ERNIE-Bot Agent 当中使用。

初始化的方式包括三种：

* `from_aistudio`: 可通过 AI Studio 工具中心获得一个 tool-id，从而进行初始化。
* `from_url`: 可通过 url 获得一个远端工具集的 `openapi.yaml` 文件，从而进行初始化。
* `from_openapi_file`: 可通过一个本地的 `openapi.yaml` 文件路径初始化，此文件包含所有远端工具的集合。

示例代码如下所示：

```python
toolkit = RemoteToolkit.from_aistudio("text-moderation")
```

### 2.4 RemoteTool

`RemoteToolkit` 是一个工具集合的概念，而 `RemoteTool` 则是特指某一个工具，对应着 openapi.yaml 中定义的某一个 api。

`RemoteTool` 负责某一个 API 的执行调用，目前仅支持 `Restful API` 的形式来调用，单独执行代码如下所示：

```python
tool = toolkit.get_tool("moderation")
result = await tool("”欢迎使用 ERNIE-Bot Agent“这句话是否含有政治敏感内容")
```

## 3. Tool 详解

LocalTool 旨在运行在本地的工具，可以是获取当前时间的函数、执行 Python 代码的Python 解释器或者执行 Shell 脚本的Shell 解释器等。

这些 LocalTool 可以不用运行在远端，而作为一个本地工具接入到智能体（Agent）当中。

为了详细的介绍，我们将以 CurrentTimeTool 为例进行详细讲解。

### 3.1 关键信息

为了确保一个工具（Tool）能够被代理（Agent）正常执行，需要进行以下配置：

1. 输入和输出参数：通过配置输入和输出参数，Agent能够从历史对话信息中提取相应的字段信息，并与工具进行有效的交互；此部分的信息通过继承 `ToolParameterView` 来实现。
2. 工具的描述信息：这部分提供了关于工具用途和功能的详细说明。Agent根据这些描述信息来判断何时应该调用该工具。描述信息应包括工具的功能、适用场景以及任何特定的调用要求；此部分的信息通过定义的 Tool 的 description 字段来实现。
3. 工具的具体逻辑：这是工具的核心部分，包含了实现特定业务逻辑的代码或指令。具体逻辑定义了工具在接收到代理的请求时应如何响应和执行相应的任务；此部分是通过继承 `__call__` 函数来实现的。
4. 工具的对话示例：这些示例用于演示工具在实际对话中的用法。通过提供对话示例，代理可以更准确地理解如何调用该工具，并在适当的情境下应用它。对话示例还可以帮助开发者测试和验证工具的有效性和正确性；此部分是通过继承 `examples` 属性来实现的。

### 3.2 CurrentTimeTool 示例

以 CurrentTimeTool 为例，下面我们将展示如何实现一个工具。首先，让我们查看整体代码结构...

```python
class CurrentTimeToolOutputView(ToolParameterView):
    current_time: str = Field(description="当前时间")


class CurrentTimeTool(Tool):
    description: str = "CurrentTimeTool 用于获取当前时间"
    ouptut_type: Type[ToolParameterView] = CurrentTimeToolOutputView

    async def __call__(self) -> Dict[str, str]:
        return {"current_time": datetime.strftime(datetime.now(), "%Y年%m月%d日 %H时%M分%S秒)}

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

## 4. 总结

以上介绍了工具的核心组成部分和如何实现一个本地 Tool，如果想要了解更多实现 LocalTool 的步骤，可参考 [创建 LocalTool](../cookbooks/agent/local_tool.ipynb)，如果想要了解更多实现 RemoteTool 的步骤，可参考 [创建 RemoteTool](../cookbooks/agent/remote_tool.ipynb)
