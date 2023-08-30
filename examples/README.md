# Gradio UI示例使用

在示例中基于Gradio提供了一个简单的Web UI Demo，开发者可以快速体验

- **ChatCompletion** 基于ernie-bot-3.5或ernie-bot-turbo的生成式对话能力
- **Embedding** 文本向量化能力，并计算不同文本间的余弦相似度
- **Image** 根据输入文本，生成不同尺寸的图像

## 环境安装

```
cd examples
pip install -r requirements.txt
```

## 使用

执行如下命令
```
python gradio_demo.py --port 8188
```

本地使用浏览器打开[http://0.0.0.0:8188](http://0.0.0.0:8188)。

### ChatCompletion

使用前，请参考[认证鉴权文档](../docs/authentication.md#%E7%94%B3%E8%AF%B7%E5%8D%83%E5%B8%86%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%B9%B3%E5%8F%B0%E7%9A%84aksk)申请千帆平台的AK/SK用于服务调用鉴权。

在打开Demo页面的左侧，在如下图所示的两处红框范围内，分别填入申请得到的AK/SK，输入对话内容提交即可

![example_pic1](https://user-images.githubusercontent.com/19339784/263580266-af87d38b-1b2e-4839-95a8-0f17678d038c.png)

### Embedding
使用前，请参考[认证鉴权文档](../docs/authentication.md#%E7%94%B3%E8%AF%B7%E5%8D%83%E5%B8%86%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%B9%B3%E5%8F%B0%E7%9A%84aksk)申请千帆平台的AK/SK用于服务调用鉴权。  
在打开Demo页面的左侧，在如下图所示的两处红框范围内，分别填入申请得到的AK/SK，输入两段文本点击提交后，Demo会展示分别计算得到的向量，以及两段文本基于向量计算得到的余弦相似度。

![example_pic2](https://user-images.githubusercontent.com/19339784/263580283-9d31a443-5bda-4258-9db7-d8e5e9f56611.png)

### Image
使用前，请参考[认证鉴权文档](../docs/authentication.md#%E7%94%B3%E8%AF%B7%E6%99%BA%E8%83%BD%E5%88%9B%E4%BD%9C%E5%B9%B3%E5%8F%B0%E7%9A%84aksk)申请智能创作平台的AK/SK用于服务调用鉴权。
![example_pic3](https://user-images.githubusercontent.com/19339784/263580304-5e1e75ce-dcf5-4b62-8b95-fe3f59be2598.png)
