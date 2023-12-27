# Remote Tool

当前企业中大部分的功能都是通过 API 的形式暴露，Agent如果想要拓展自己的能力边界，就必须基于现有的功能性 API（eg：查天气或查火车票的 api）来进行交互，从而实现更复杂的企业级功能。

而Agent想与存量功能性 API 进行交互需要有一个标准的交互协议，而ErnieBot-Agent 中已经提供了 RemoteTool 和 RemoteToolkit 来简化此交互流程，接下来将介绍 如何在 ErnieBot-Agent 中使用 RemoteTool。

## 1. 使用 RemoteTool

RemoteTool（远程工具）可以是开发者自己提供的，也可以上 AI Studio 的工具中心搜索对应工具，使用代码如下所示：

### 1.1 开发者提供的RemoteTool

```python
toolkit = RemoteToolkit.from_url("http://xxx.com")  # 必须存在：http://xxx.com/.well-known/openapi.yaml
agent = FunctionalAgent(llm, tools=toolkit.get_tools())
```

### 1.2 AI Studio 工具中心

```python
toolkit = RemoteToolkit.from_aistudio("translation")
agent = FunctionalAgent(llm, tools=toolkit.get_tools())
```

使用起来是不是很简单呢？可是其中的原理是如何呢？

## 2. RemoteTool vs RemoteToolkit

RemoteTool 是单个远程工具，比如添加单词到单词本功能属于单个 RemoteTool，可是：添加单词、删除单词和查询单词这几个功能组装在一起就组成了一个 Toolkit（工具箱），并且也只在远程工具中存在：通常一个 url 下会暴露出多个 API，此时可通过一个配置文件暴露多个 Tool。

以下将会统一使用 RemoteTool 来标识远程工具。

## 3. RemoteTool 如何与 Agent 交互

无论是 LocalTool 还是 RemoteTool 都必须要提供核心的信息：

* tool 的描述信息
* tool 的输入和输出参数
* tool 的执行示例

LocalTool 是通过代码定义上述信息，而 RemoteTool 则是通过`openapi.yaml`来定义上述信息，RemoteToolkit 在加载时将会解析`openapi.yaml`中的信息，并在执行时将对应 Tool 的元信息传入 Agent LLM 当中。

此外 RemoteTool 的远端调用是通过 http 的方式执行，同时遵照 [OpenAPI 3.0](https://swagger.io/specification/) 的规范发送请求并解析响应。

## 4. RemoteTool Server

* server 端代码

```python
@app.post("/add-word")
def add_word():
    word = request.json()["word"]
    ...
    return jsonify({"result": "..."})

@app.get("/.well-known/openapi.yaml")
def get_openapi():
    return send_file(".well-known/openapi.yaml")
```

* openapi.yaml

```yaml
openapi: 3.0.1
info:
    title: 单词本
    description: 个性化的英文单词本，可以增加、删除和浏览单词本中的单词，背单词时从已有单词本中随机抽取单词生成句子或者段落。
    version: "v1"
servers:
    - url: http://127.0.0.1:8081
paths:
    /add_word:
        post:
            operationId: addWord
            description: 在单词本中添加一个单词
            requestBody:
                required: true
                content:
                    application/json:
                        schema:
                            $ref: "#/components/schemas/addWord"
            responses:
                "200":
                    description: 单词添加成功
                    content:
                        application/json:
                            schema:
                                $ref: "#/components/schemas/messageResponse"

components:
    schemas:
        addWord:
            type: object
            required: [word]
            properties:
                word:
                    type: string
                    description: 需要添加到单词本中的一个单词
        messageResponse:
            type: object
            required: [message]
            properties:
                result:
                    type: string
                    description: 回复信息
```

* 调用远端代码

```python
toolkit = RemoteToolkit.from_url("http://127.0.0.1:5000")
agent = FunctionalAgent(llm, tools=toolkit.get_tools())
```

> 详细可参考：[如何从0到1开发自己的插件](https://yiyan.baidu.com/developer/doc#5llaiqbti)，此处的插件就是 RemoteTool。
