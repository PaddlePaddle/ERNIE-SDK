# 认证鉴权

调用文心大模型是一项收费服务，所以大家使用ERNIE Bot SDK需要进行认证鉴权。

认证鉴权主要包括两步：申请用户凭证，设置鉴权参数。

ERNIE Bot SDK支持多个后端来调用文心大模型（如下表格），大家可以根据实际情况选择。不同后端支持的模型、认证鉴权方式存在差异，下面我们分别介绍。

| 后端   |  API_TYPE  |  支持的模型 |
| :---     | :----      |   :---  |
| AI Studio | aistudio |  ernie-bot，ernie-bot-turbo，ernie-text-embedding |
| 千帆大模型平台 | qianfan |  ernie-bot，ernie-bot-turbo，ernie-text-embedding |
| 智能创作平台 | yinian |  ernie-vilg-v2 |

## 1 `AI Studio`后端的认证鉴权

### 1.1 申请用户凭证

在[AI Studio星河社区](https://aistudio.baidu.com/index)注册并登录账号，可以在个人中心的[访问令牌页面](https://aistudio.baidu.com/usercenter/token)获取用户凭证access token。

<div align="center">
<img src="https://user-images.githubusercontent.com/52520497/268609784-8476269e-5cdb-4dfc-9841-983b5a766226.png" width="800">  
</div>

注意事项：

* AI Studio每个账户的access token，有100万token的免费额度，可以用于ERNIE Bot SDK调用文心一言大模型。AI Studio近期将会开通付费购买的渠道。
* access token是私密信息，切记不要对外公开。

### 1.2 设置鉴权参数

AI Studio后端可以使用access token进行鉴权，支持如下三种方法来设置鉴权参数。

(1) 使用环境变量：

``` {.copy}
export EB_API_TYPE="aistudio"
export EB_ACCESS_TOKEN="<access-token-for-aistudio>"
```

(2) 使用全局变量：

``` {.py .copy}
import erniebot

erniebot.api_type = "aistudio"
erniebot.access_token = "<access-token-for-aistudio>"
```

(3) 使用`_config_`参数：

``` {.py .copy}
import erniebot

response = erniebot.ChatCompletion.create(
    _config_=dict(
        api_type="aistudio",
        access_token="<access-token-for-aistudio>",
    ),
    model="ernie-bot",
    messages=[{"role": "user", "content": "你好，请介绍下你自己",
    }],
)
```

如果大家同时使用多种方式设置鉴权参数，ERNIE Bot SDK将根据优先级确定配置项的最终取值（其他后端类似）。三种设置方式的优先级从高到低依次为：使用`_config_`参数 > 使用全局变量 > 使用环境变量。

## 2 `千帆大模型平台`后端的认证鉴权

### 2.1 申请用户凭证

申请流程：

* 进入[百度云](https://cloud.baidu.com/)，完成注册。
* 进入百度云 - [千帆大模型平台](https://cloud.baidu.com/product/wenxinworkshop)，提交申请体验。通常几个小时后会通知申请通过。
* 登录[千帆大模型平台](https://cloud.baidu.com/product/wenxinworkshop)，进入[控制台](https://console.bce.baidu.com/qianfan/ais/console/applicationConsole/application)创建千帆应用，可以拿到AK/SK（如下图）。
* 进入[计费管理](https://console.bce.baidu.com/qianfan/chargemanage/list)，选择服务并开通付费，包括：`ERNIE-Bot大模型公有云在线调用服务`、`ERNIE-Bot-turbo大模型公有云在线调用服务`和`Embedding-V1公有云在线调用服务`。
* 参考[access token获取教程](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Ilkkrb0i5)，使用AK/SK获取access token。

<div align="center">
<img src="https://user-images.githubusercontent.com/52520497/264009567-46f88a38-df70-4a79-affb-ddbf797855b1.jpeg" width="800">  
</div>

注意事项：
* 千帆的AK/SK是一直有效，而access token的默认有效期是30天，失效后需要重新获取access token。

### 2.2 设置鉴权参数

千帆后端也可以使用access token进行鉴权，设置鉴权参数的三种方法和AI Studio后端类似，举例如下。

请注意设置后端参数为`'qianfan'`，并且使用千帆平台申请的access token。

``` {.copy}
export EB_API_TYPE="qianfan"
export EB_ACCESS_TOKEN="<access-token-for-qianfan>"
```

``` {.py .copy}
import erniebot

erniebot.api_type = "qianfan"
erniebot.access_token = "<access-token-for-qianfan>"
```

此外，千帆后端还可以使用AK/SK进行鉴权，同样支持三种方法，环境变量对应是`EB_AK`和`EB_SK`，Python变量对应是`ak`和`sk`，举例如下。

``` {.copy}
export EB_API_TYPE="qianfan"
export EB_AK="<access-key-for-qianfan>"
export EB_SK="<secret-access-key-for-qianfan>"
```

``` {.py .copy}
import erniebot

erniebot.api_type = "qianfan"
erniebot.ak = "<access-key-for-qianfan>"
erniebot.sk = "<secret-access-key-for-qianfan>"
```

## 3 `智能创作平台`后端的认证鉴权

### 3.1 申请用户凭证

申请流程：

* 进入[百度云](https://cloud.baidu.com/)，完成注册。
* 进入百度云 - [智能创作平台](https://console.bce.baidu.com/ai/#/ai/intelligentwriting/app/list)，创建应用，可以拿到AK/SK（如下图）。
* 参考[access token获取教程](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Ilkkrb0i5)，使用AK和SK获取access token。

<div align="center">
<img src="https://user-images.githubusercontent.com/52520497/264009612-17658684-c066-44e5-8814-178214aa8155.jpeg" width="800">  
</div>

智能创作平台的完整介绍，请参考[使用文档](https://ai.baidu.com/ai-doc/NLP/Uk53wndcb)；费用、充值相关的问题，请参考[计费简介](https://ai.baidu.com/ai-doc/NLP/qla2beec2)。

### 3.2 设置鉴权参数

智能创作平台后端设置鉴权参数的方法，和千帆后端完全一致，都支持access toke或者AK/SK参数，举例如下。

请注意设置后端参数为`'yinian'`，并且使用智能创作平台申请的access_token、AK/SK。

使用access token的例子：

``` {.copy}
export EB_API_TYPE="yinian"
export EB_ACCESS_TOKEN="<access-token-for-yinian>"
```

``` {.py .copy}
import erniebot

erniebot.api_type = "yinian"
erniebot.access_token = "<access-token-for-yinian>"
```

使用AK/SK的例子：

``` {.copy}
export EB_API_TYPE="yinian"
export EB_AK="<access-key-for-yinian>"
export EB_SK="<secret-access-key-for-yinian>"
```

``` {.py .copy}
import erniebot

erniebot.api_type = "yinian"
erniebot.ak = "<access-key-for-yinian>"
erniebot.sk = "<secret-access-key-for-yinian>"
```
