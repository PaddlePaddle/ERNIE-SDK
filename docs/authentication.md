# 认证鉴权

## 认证鉴权

大家使用ERNIE Bot SDK，首先需要申请AK/SK，然后设置认证鉴权的参数，最后使用接口调用文心大模型的能力。

ERNIE Bot SDK支持的文心大模型来自多个后端平台，大家可以阅读下表，参照文档申请AK/SK。

| 后端平台   |  API_TYPE  |  支持模型 | 申请AK/SK的方法 |
| :---     | :----      | :----  | :----  |
| 千帆大模型平台 | qianfan | ernie-bot-3.5, ernie-bot-turbo, ernie-text-embedding | [申请千帆大模型平台的AK/SK](#申请千帆大模型平台的aksk)
| 智能创作平台 | yinian | ernie-vilg-v2 | [申请智能创作平台的AK/SK](#申请智能创作平台的aksk)


ERNIE Bot SDK认证鉴权需要设置三个输入参数：

* 后端平台，支持`qianfan`和`yinian`，默认是`qianfan`。
* 后端平台上申请的API Key。
* 后端平台上申请的Secret Key。


ERNIE Bot SDK支持3种方式设置认证鉴权的参数，大家可以自由选择。

1）使用环境变量：
```shell
export EB_API_TYPE="<EB-API-TYPE>"
export EB_AK="<EB-ACCESS-KEY>"
export EB_SK="<EB-SECRET-KEY>"
```

2）使用全局变量：
``` {.py .copy}
import erniebot

erniebot.api_type = "<EB-API-TYPE>"
erniebot.ak = "<EB-ACCESS-KEY>"
erniebot.sk = "<EB-SECRET-KEY>"
```

3) 使用`_config_`参数：
``` {.py .copy}
import erniebot

# Take erniebot.ChatCompletion as an example
chat_completion = erniebot.ChatCompletion.create(
    _config_=dict(
        api_type="<EB-API-TYPE>",
        ak="<EB-ACCESS-KEY>",
        sk="<EB-SECRET-KEY>",
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
* **使用特定模型，请准确设置对应后端平台的认证鉴权参数。**

## 申请千帆大模型平台的AK/SK

具体流程：

* 进入[百度云](https://cloud.baidu.com/)，完成注册。
* 进入百度云 - [千帆大模型平台](https://cloud.baidu.com/product/wenxinworkshop)，提交申请体验。通常几个小时后会通知申请通过。
* 登录[千帆大模型平台](https://cloud.baidu.com/product/wenxinworkshop)，进入[控制台](https://console.bce.baidu.com/qianfan/ais/console/applicationConsole/application)创建千帆应用，可以拿到AK/SK（如下图）。
* 进入[计费管理](https://console.bce.baidu.com/qianfan/chargemanage/list)，选择服务并开通付费，比如：`ERNIE-Bot大模型公有云在线调用服务`、`ERNIE-Bot-turbo大模型公有云在线调用服务`和`Embedding-V1公有云在线调用服务`。

<div align="center">
<img src="https://user-images.githubusercontent.com/52520497/264009567-46f88a38-df70-4a79-affb-ddbf797855b1.jpeg"  width = "800" />  
</div>

注意事项：

* AK/SK是私人信息，大家不要分享给他人，也不要对外暴露。
* 千帆大模型平台的新用户，默认会有20元代金券，大家可以快速体验ERNIE Bot SDK。
* 千帆大模型平台的完整介绍，请参考[使用文档](https://cloud.baidu.com/doc/WENXINWORKSHOP/index.html)；费用、充值相关的问题，请参考[价格文档](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Blfmc9dlf)

## 申请智能创作平台的AK/SK

具体流程：

* 进入[百度云](https://cloud.baidu.com/)，完成注册。
* 进入百度云 - [智能创作平台](https://console.bce.baidu.com/ai/#/ai/intelligentwriting/app/list)，创建应用，可以拿到AK/SK（如下图）。

<div align="center">
<img src="https://user-images.githubusercontent.com/52520497/264009612-17658684-c066-44e5-8814-178214aa8155.jpeg"  width = "800" />  
</div>

注意事项：

* AK/SK是私人信息，大家不要分享给他人，也不要对外暴露。
* 智能创作平台的完整介绍，请参考[使用文档](https://ai.baidu.com/ai-doc/NLP/Uk53wndcb)；费用、充值相关的问题，请参考[计费简介](https://ai.baidu.com/ai-doc/NLP/qla2beec2)。
