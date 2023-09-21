# 基础功能演示应用

基础功能演示应用提供对话补全、语义向量以及文生图demo，可供大家快速体验ERNIE Bot SDK的功能。

* **对话补全（Chat Completion）** 基于ernie-bot或ernie-bot-turbo的生成式对话能力；
* **语义向量（Embedding）** 提取输入文本的语义向量表征，并计算不同向量间的余弦相似度；
* **文生图（Image Generation）** 根据输入文本，生成不同尺寸的图像。

## 安装依赖

首先，请确保Python版本>=3.8。然后执行如下命令：

```shell
cd examples/quick_start
pip install -r requirements.txt
```

## 运行应用

执行如下命令：

```shell
python gradio_demo.py
```

然后，使用本地浏览器打开[http://localhost:8073](http://localhost:8073)。

可以通过`--port`指定端口号，例如`python gradio_demo.py --port 8188`指定8188端口作为启动Gradio服务的端口。默认端口为8073。

### 对话补全（Chat Completion）

使用前，请参考[认证鉴权文档](../../docs/authentication.md)获取要使用的后端的鉴权参数。

在页面的左侧如下图所示的两处红框处分别填入申请得到的AK/SK，之后输入对话内容并点击提交即可。

![example_pic1](https://user-images.githubusercontent.com/19339784/263580266-af87d38b-1b2e-4839-95a8-0f17678d038c.png)

### 语义向量（Embedding）

使用前，请参考[认证鉴权文档](../../docs/authentication.md)获取要使用的后端的鉴权参数。

在页面左侧如下图所示的两处红框处分别填入申请得到的AK/SK。输入两段文本，点击提交后，demo会展示提取到的向量，以及计算得到的两个向量间的余弦相似度。

![example_pic2](https://user-images.githubusercontent.com/19339784/263580283-9d31a443-5bda-4258-9db7-d8e5e9f56611.png)

### 文生图（Image Generation）

使用前，请参考[认证鉴权文档](../../docs/authentication.md)申请智能创作平台的AK/SK用于服务调用鉴权。

![example_pic3](https://user-images.githubusercontent.com/19339784/263580304-5e1e75ce-dcf5-4b62-8b95-fe3f59be2598.png)
