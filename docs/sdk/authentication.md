# 认证鉴权

调用文心大模型可能涉及到收费服务，所以大家使用ERNIE Bot时需要进行认证鉴权。认证鉴权主要包括两步：申请用户凭证，设置鉴权参数。

ERNIE Bot支持多个后端平台来调用文心大模型（如下表格），大家可以根据实际情况选择。不同后端平台支持的模型、认证鉴权方式存在差异，下面我们分别介绍。

| 后端 | api_type | 支持的模型 |
| :--- | :--- | :--- |
| AI Studio | aistudio | ernie-3.5，ernie-turbo，ernie-4.0，ernie-longtext，ernie-text-embedding |
| 千帆大模型平台 | qianfan | ernie-3.5，ernie-turbo，ernie-4.0，ernie-longtext，ernie-text-embedding |
| 智能创作平台 | yinian | ernie-vilg-v2 |

## AI Studio后端的认证鉴权

### 申请用户凭证

在[AI Studio星河社区](https://aistudio.baidu.com/index)注册并登录账号，可以在个人中心的[访问令牌页面](https://aistudio.baidu.com/usercenter/token)获取access token。

<div align="center">
<img src="https://user-images.githubusercontent.com/52520497/268609784-8476269e-5cdb-4dfc-9841-983b5a766226.png" width="800" />  
</div>

注意事项：

* AI Studio每个账户的access token，有100万token的免费额度，可以用于ERNIE Bot调用文心大模型。
* 在[token管理页面](https://aistudio.baidu.com/token/manage)可以查看token获取、消耗明细和过期记录，或者购买更多token。
* access token是私密信息，切记不要对外公开。

### 设置鉴权参数

AI Studio后端可以使用access token进行鉴权，支持如下三种方法来设置鉴权参数。

1. 使用环境变量：

    ```{.sh .copy}
    export EB_API_TYPE="aistudio"
    export EB_ACCESS_TOKEN="<access-token-for-aistudio>"
    ```

2. 使用全局变量：

    ```{.py .copy}
    import erniebot

    erniebot.api_type = "aistudio"
    erniebot.access_token = "<access-token-for-aistudio>"
    ```

3. 使用`_config_`参数：

    ```{.py .copy}
    import erniebot

    response = erniebot.ChatCompletion.create(
        _config_=dict(
            api_type="aistudio",
            access_token="<access-token-for-aistudio>",
        ),
        model="ernie-3.5",
        messages=[{"role": "user", "content": "你好，请介绍下你自己",
        }],
    )
    ```

如果同时使用多种方式设置鉴权参数，ERNIE Bot将根据优先级确定配置项的最终取值（其他后端类似）。三种设置方式的优先级从高到低依次为：使用`_config_`参数 > 使用全局变量 > 使用环境变量。

## 千帆大模型平台后端的认证鉴权

### 申请用户凭证

申请流程：

1. 进入[百度云官网](https://cloud.baidu.com/)，完成注册。
2. 进入百度云 - [千帆大模型平台](https://cloud.baidu.com/product/wenxinworkshop)，提交体验申请。通常几个小时后会通知申请通过。
3. 登录[千帆大模型平台](https://cloud.baidu.com/product/wenxinworkshop)，进入[控制台](https://console.bce.baidu.com/qianfan/ais/console/applicationConsole/application)创建千帆应用，可以获取到API key与secret key（如下图）。
4. 进入[计费管理页面](https://console.bce.baidu.com/qianfan/chargemanage/list)，选择需要用到的服务并开通付费，例如：ERNIE-Bot大模型公有云在线调用服务、ERNIE-Bot-turbo大模型公有云在线调用服务、Embedding-V1公有云在线调用服务、ERNIE-Bot 4.0大模型公有云在线调用服务等。
5. （非必需）参考[access token获取教程](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Ilkkrb0i5)，使用API key和secret key获取access token。

<div align="center">
<img src="https://user-images.githubusercontent.com/52520497/264009567-46f88a38-df70-4a79-affb-ddbf797855b1.jpeg" width="800" />  
</div>

注意事项：

* 千帆的API key和secret key长期有效，而access token的默认有效期是30天，失效后需要重新获取access token。

### 设置鉴权参数

千帆后端可以使用access token或者API key+secret key进行鉴权。

1. 使用access token进行鉴权。为千帆后端设置鉴权参数的三种方法和AI Studio后端类似，举例如下。

    请注意设置后端参数为`"qianfan"`，并且使用千帆平台申请的access token。

    ```{.sh .copy}
    export EB_API_TYPE="qianfan"
    export EB_ACCESS_TOKEN="<access-token-for-qianfan>"
    ```

    ```{.py .copy}
    import erniebot

    erniebot.api_type = "qianfan"
    erniebot.access_token = "<access-token-for-qianfan>"
    ```

2. 使用API key与secret key进行鉴权。千帆后端同样支持三种参数配置方法，与API key和secret key对应的环境变量是`EB_AK`和`EB_SK`、Python变量是`ak`和`sk`。举例如下：

    ```{.sh .copy}
    export EB_API_TYPE="qianfan"
    export EB_AK="<api-key-for-qianfan>"
    export EB_SK="<secret-key-for-qianfan>"
    ```

    ```{.py .copy}
    import erniebot

    erniebot.api_type = "qianfan"
    erniebot.ak = "<api-key-for-qianfan>"
    erniebot.sk = "<secret-key-for-qianfan>"
    ```

## 智能创作平台后端的认证鉴权

### 申请用户凭证

申请流程：

1. 进入[百度云官网](https://cloud.baidu.com/)，完成注册。
2. 进入百度云 - 智能创作平台 - [应用页面](https://console.bce.baidu.com/ai/#/ai/intelligentwriting/app/list)，创建应用，可以拿到API key和secret key（如下图）。

    <div align="center">
    <img src="https://user-images.githubusercontent.com/52520497/264009612-17658684-c066-44e5-8814-178214aa8155.jpeg" width="800" />  
    </div>

3. 进入百度云 - 智能创作平台 - [概览页面](https://console.bce.baidu.com/ai/#/ai/intelligentwriting/overview/index)，在服务列表中找到AI作画-高级版服务并开通付费（如下图）。

    <div align="center">
    <img src="https://github.com/PaddlePaddle/PaddleSeg/assets/52520497/7c855314-8332-47ad-a444-a08dd37ec32f" width="800" />  
    </div>

4. （非必需）参考[access token获取教程](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Ilkkrb0i5)，使用API key和secret key获取access token。

智能创作平台的完整介绍，请参考[使用文档](https://ai.baidu.com/ai-doc/NLP/Uk53wndcb)；费用、充值相关的问题，请参考[计费简介](https://ai.baidu.com/ai-doc/NLP/qla2beec2)。

### 设置鉴权参数

为智能创作平台后端设置鉴权参数的方法，和千帆后端完全一致，都支持access token或者API key+secret key。举例如下。

请注意设置后端参数为`"yinian"`，并且使用智能创作平台申请的access token、API key、secret key。

1. 使用access token的例子：

    ```{.sh .copy}
    export EB_API_TYPE="yinian"
    export EB_ACCESS_TOKEN="<access-token-for-yinian>"
    ```

    ```{.py .copy}
    import erniebot

    erniebot.api_type = "yinian"
    erniebot.access_token = "<access-token-for-yinian>"
    ```

2. 使用API key和secret key的例子：

    ```{.sh .copy}
    export EB_API_TYPE="yinian"
    export EB_AK="<api-key-for-yinian>"
    export EB_SK="<secret-key-for-yinian>"
    ```

    ```{.py .copy}
    import erniebot

    erniebot.api_type = "yinian"
    erniebot.ak = "<api-key-for-yinian>"
    erniebot.sk = "<secret-key-for-yinian>"
    ```
