# 文生图（Image Generation）

## 介绍

文生图是指根据文本提示、图像尺寸等信息，使用文心大模型，自动创作图片。

目前文心提供如下向量表示模型：

| 模型 | 说明 | API调用方式 |
| :--- | :---- | :----- |
| ernie-text-embedding | 支持计算最多384个token的文本的向量表示。 | `erniebot.Image.create(model='ernie-vilg-v2', ...)` |

参阅[Image API文档](../api_reference/image.md)了解API的完整使用方式。

请注意，目前仅`yinian`后端支持文生图功能。

## 使用示例

大家可以使用下面示例代码，体验文生图功能（请注意替换成自己的access token）。

执行完成后，请及时点击链接下载创作的图片，默认1小时后链接失效。

```{.py .copy}
import erniebot

erniebot.api_type = 'yinian'
erniebot.access_token = '<access-token-for-yinian>'

response = erniebot.Image.create(model='ernie-vilg-v2', prompt="请帮我画一只可爱的大猫咪", width=512, height=512, version='v2', image_num=1)

print(response.get_result())
```

文本提示是“请帮我画一只可爱的大猫咪”时，生成的图片如下：

<div align="center">
<img src="https://user-images.githubusercontent.com/52520497/263970054-abf68cb8-3ad3-48cb-942f-1fc3075d5452.png" width="400">  
</div>

文本提示是“请帮我画一只开心的袋熊”时，生成的图片如下：

<div align="center">
<img src="https://user-images.githubusercontent.com/52520497/263970013-53eef22c-5ad0-4d60-835b-5f7b699fb3ef.png" width="400">  
</div>
