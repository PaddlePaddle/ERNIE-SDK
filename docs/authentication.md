# 认证鉴权

## 认证鉴权

在使用ERNIE Bot SDK提供的接口调用文心大模型的能力前，首先需要申请鉴权参数，然后设置鉴权参数。

目前ERNIE Bot SDK支持的鉴权参数如下：

| API参数名称   | 环境变量名称  |  类型   |  必须设置 |  描述   |
| :---         | :----       | :----  | :---- |  :---- |
| api_type     | EB_API_TYPE | string | 否 | 设置后端平台的类型。支持`'qianfan'`和`'yinian'`，默认是`'qianfan'`。|
| ak           | EB_AK       | string | 否 | 设置认证鉴权的access key。必须和`sk`同时设置。 |
| sk           | EB_SK       | string | 否 | 设置认证鉴权的secret key。必须和`ak`同时设置。 |
| access_token | EB_ACCESS_TOKEN | string | 否 | 设置认证鉴权的access token。推荐优先使用`ak`和`sk`。如果设置了`access_token`，则使用该access token；如果`access_token`没有设置或者失效，并且设置了`ak`和`sk`，部分后端平台类型支持自动通过`ak`和`sk`获取access token。|
| access_token_path | EB_ACCESS_TOKEN_PATH | string | 否 | 设置存有access token的文件路径。推荐优先使用`ak`和`sk`。`access_token_path`生效原理和`access_token`相同。|

ERNIE Bot SDK支持的文心大模型来自多个后端平台，不同平台支持的鉴权参数不尽相同。请阅读下表，参照对应的文档申请鉴权参数。

| 后端平台   |  API_TYPE  |  支持模型 | 申请鉴权参数的方法 | 是否支持AK/SK |
| :---     | :----      | :----  | :----  | :---  |
| 千帆大模型平台 | `qianfan` | `ernie-bot-3.5`, `ernie-bot-turbo`, `ernie-text-embedding` | [申请千帆大模型平台的鉴权参数](#申请千帆大模型平台的鉴权参数)| 是 |
| 智能创作平台 | `yinian` | `ernie-vilg-v2` | [申请智能创作平台的鉴权参数](#申请智能创作平台的鉴权参数) | 是 |
| AI Studio | `ai_studio` | `ernie-bot-3.5`, `ernie-bot-turbo`, `ernie-text-embedding` | [申请AI Studio平台的鉴权参数](#申请ai-studio平台的鉴权参数) | 否 |

与其它参数类似，鉴权参数可通过如下3种方式设置，请根据需要自由选择。关于参数配置的更多技巧，请在[此文档](./configuration.md)了解。

1）使用环境变量：
```shell
export EB_API_TYPE="<EB-API-TYPE>"
export EB_AK="<EB-ACCESS-KEY>"
export EB_SK="<EB-SECRET-KEY>"
export EB_ACCESS_TOKEN="<EB-ACCESS-TOKEN>"
```

2）使用全局变量：
``` {.py .copy}
import erniebot

erniebot.api_type = "<EB-API-TYPE>"
erniebot.ak = "<EB-ACCESS-KEY>"
erniebot.sk = "<EB-SECRET-KEY>"
erniebot.access_token = "<EB-ACCESS-TOKEN>"
```

3) 使用`_config_`参数：
``` {.py .copy}
import erniebot

chat_completion = erniebot.ChatCompletion.create(
    _config_=dict(
        api_type="<EB-API-TYPE>",
        ak="<EB-ACCESS-KEY>",
        sk="<EB-SECRET-KEY>",
        access_token="<EB-ACCESS-TOKEN>",
    ),
    model="ernie-bot-3.5",
    messages=[{
        "role": "user",
        "content": "你好，请介绍下你自己",
    }],
)
```

注意事项：

* 允许同时使用多种方式设置鉴权信息，程序将根据设置方式的优先级确定配置项的最终取值。三种设置方式的优先级从高到低依次为：使用`_config_`参数，使用全局变量，使用环境变量。
* **使用特定模型，请准确设置对应后端平台的鉴权参数。**
    * 对于所有后端平台，均支持仅指定access token、而不指定其余的鉴权参数。
    * 对于支持AK/SK的后端平台，允许仅设置AK和SK、而将access token留空，在这种情况下程序将根据AK和SK自动维护access token。
    * 对于不支持AK/SK的后端平台，只需设定access token。

## 申请千帆大模型平台的鉴权参数

具体流程：

* 进入[百度云](https://cloud.baidu.com/)，完成注册。
* 进入百度云 - [千帆大模型平台](https://cloud.baidu.com/product/wenxinworkshop)，提交申请体验。通常几个小时后会通知申请通过。
* 登录[千帆大模型平台](https://cloud.baidu.com/product/wenxinworkshop)，进入[控制台](https://console.bce.baidu.com/qianfan/ais/console/applicationConsole/application)创建千帆应用，可以拿到AK/SK（如下图）。
* 进入[计费管理](https://console.bce.baidu.com/qianfan/chargemanage/list)，选择服务并开通付费，比如：`ERNIE-Bot大模型公有云在线调用服务`、`ERNIE-Bot-turbo大模型公有云在线调用服务`和`Embedding-V1公有云在线调用服务`。
* （可选）参考[access token获取教程](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Ilkkrb0i5)，使用AK和SK获取access token。

<div align="center">
<img src="https://user-images.githubusercontent.com/52520497/264009567-46f88a38-df70-4a79-affb-ddbf797855b1.jpeg"  width = "800" />  
</div>

注意事项：

* AK/SK是私人信息，大家不要分享给他人，也不要对外暴露。
* 千帆大模型平台的新用户，默认会有20元代金券，大家可以快速体验ERNIE Bot SDK。
* 千帆大模型平台的完整介绍，请参考[使用文档](https://cloud.baidu.com/doc/WENXINWORKSHOP/index.html)；费用、充值相关的问题，请参考[价格文档](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Blfmc9dlf)

## 申请智能创作平台的鉴权参数

具体流程：

* 进入[百度云](https://cloud.baidu.com/)，完成注册。
* 进入百度云 - [智能创作平台](https://console.bce.baidu.com/ai/#/ai/intelligentwriting/app/list)，创建应用，可以拿到AK/SK（如下图）。
* （可选）参考[access token获取教程](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Ilkkrb0i5)，使用AK和SK获取access token。

<div align="center">
<img src="https://user-images.githubusercontent.com/52520497/264009612-17658684-c066-44e5-8814-178214aa8155.jpeg"  width = "800" />  
</div>

注意事项：

* AK/SK是私人信息，大家不要分享给他人，也不要对外暴露。
* 智能创作平台的完整介绍，请参考[使用文档](https://ai.baidu.com/ai-doc/NLP/Uk53wndcb)；费用、充值相关的问题，请参考[计费简介](https://ai.baidu.com/ai-doc/NLP/qla2beec2)。

## 申请AI Studio平台的鉴权参数
