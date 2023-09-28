# Image

根据文本提示、尺寸等信息，使用文心一格大模型，自动创作图片。

## 函数接口

```{.py .copy}
erniebot.Image.create(**kwargs: Any) -> EBResponse:
```
## 输入参数

调用Image接口前，需要首先设置`api_type`参数。

```{.py .copy}
ernie.api_type = "yinian"
```

`erniebot.Image.create` 接口的详细参数如下：

| 参数名 | 类型     | 必填    | 描述   |
| :-----| :-----  | :----- | :----- |
| model | string  | 是 | 模型名称。当前支持`'ernie-vilg-v2'`。|
| version | string | 否 | 模型版本。支持`'v1'`、`'v2'`，分别对应模型的v1和v2版本，默认为`'v2'`。v2为最新模型，比v1在准确度、精细度上有比较明显的提升，且v2支持更多尺寸。 |
| prompt | string | 是 | 生图的文本描述。仅支持中文、日常标点符号。不支持英文，特殊符号，限制200字。 |
| width | int     | 是 | 图片宽度。v1版本支持的图像尺寸有：1024x1024、1280x720、720x1280、2048x2048、2560x1440、1440x2560；v2版本支持的图像尺寸有：512x512、640x360、360x640、1024x1024、1280x720、720x1280、2048x2048、2560x1440、1440x2560。 |
| height | int     | 是 | 图片高度。v1版本支持的图像尺寸有：1024x1024、1280x720、720x1280、2048x2048、2560x1440、1440x2560；v2版本支持的图像尺寸有：512x512、640x360、360x640、1024x1024、1280x720、720x1280、2048x2048、2560x1440、1440x2560。 |
| image_num | int | 否 | 生成图片数量。默认一张，支持生成1-8张。 |

## 返回结果

接口返回`erniebot.response.EBResponse`对象。

返回结果的一个典型示例如下：

```python
{
   "data": {
     "task_id": 1659384536691865192,
     "task_status": "SUCCESS",
     "task_progress": 1
     "sub_task_result_list": [
        {
          "sub_task_status": "SUCCESS",
          "sub_task_progress": 1,
          "sub_task_error_code": 0,
          "final_image_list": [
           {  
              "img_url": "http://aigc-t2p.bj.bcebos.com/artist-long/_final.png?02d252c87b91ed3b2f6327db0",
              "width": 512,
              "height": 512,
              "img_approve_conclusion": "pass"
            }
        ]
      }
    ]
  }
}
```

返回结果的具体字段含义如下表：

| 字段名  | 类型   | 描述  |
| :--- | :---- | :---- |
| code | int | 请求返回状态。 |
| data | dict | 返回数据。 |

`data`包含如下键值对：

| 键名  | 值类型   | 值描述  |
| :--- | :---- | :---- |
| task_id | int | 任务ID。 |
| task_status | string | 任务总体状态，有`'INIT'`（初始化）、`'WAIT'`（排队中）、`'RUNNING'`（生成中）、`'FAILED'`（失败）、`'SUCCESS'`（成功）五种状态，只有`'SUCCESS'`为成功状态。 |
| task_progress | int | 任务总体进度，`0`表示未处理完，`1`表示处理完成。 |
| sub_task_result_list | list[dict] | 子任务的结果列表。 |

`sub_task_result_list`为一个Python list，其中每个元素为一个dict，包含如下键值对：

| 键名  | 值类型   | 值描述  |
| :--- | :---- | :---- |
| sub_task_status | string | 子任务状态，有`'INIT'`（初始化）、`'WAIT'`（排队中）、`'RUNNING'`（生成中）、`'FAILED'`（失败）、`'SUCCESS'`（成功）五种状态，只有`'SUCCESS'`为成功状态。 |
| sub_task_progress | int | 子任务进度，`0`表示未处理完，`1`表示处理完成。 |
| sub_task_error_code | int | 子任务任务错误码，`0`表示正常，`501`表示文本黄反拦截，`201`表示模型生图失败。 |
| final_image_list | list[dict] | 子任务生成图像的列表。列表中的元素均为dict，包含如下键值对：<br>`img_url`：图片的下载地址，默认1小时后失效； <br>`height`：图片的高度； <br>`width`：图片的宽度； <br>`img_approve_conclusion`：图片机审结果，`'block'`表示输出图片违规，`'review'`表示输出图片疑似违规，`'pass'`表示输出图片未发现问题。 |

假设`resp`为一个`erniebot.response.EBResponse`对象，字段的访问方式有2种：`resp['data']`或`resp.data`均可获取`data`字段的内容。此外，使用`resp.get_result()`可以获取响应中的“主要结果”。对于此接口来说，`resp.get_result()`的返回结果与`resp.data`一致。

## 使用示例

```{.py .copy}
import erniebot

erniebot.api_type = "yinian
erniebot.access_token = "<access-token-for-yinian>"

response = erniebot.Image.create(model="ernie-vilg-v2", prompt="请帮我画一只可爱的大猫咪", width=512, height=512, version="v2", image_num=1)

print(response.result())
```
