# 参数配置

ERNIE Bot SDK参数配置，主要涉及认证鉴权、后端平台等信息。

ERNIE Bot SDK支持3种参数配置的方法：（1）使用环境变量，（2）使用全局变量，（3）使用`_config_`参数。

1. 使用环境变量：

    ```{.sh .copy}
    export EB_API_TYPE="<eb-api-type>"
    ```

2. 使用全局变量：

    ```{.py .copy}
    import erniebot
    erniebot.api_type = "<eb-api-type>"
    ```

3. 使用`_config_`参数：

    ```{.py .copy}
    import erniebot

    response = erniebot.ChatCompletion.create(
        _config_=dict(
            api_type="<eb-api-type>",
        ),
        model="ernie-bot",
        messages=[{
            "role": "user",
            "content": "你好，请介绍下你自己",
        }],
    )
    ```

注意：允许同时使用多种方式设置鉴权信息，程序将根据设置方式的优先级确定配置项的最终取值。三种设置方式的优先级从高到低依次为：使用`_config_`参数，使用全局变量，使用环境变量。

ERNIE Bot SDK支持的参数，具体介绍如下：

| API参数名称 | 环境变量名称 | 类型 | 必须设置 | 描述 |
| :--- | :--- | :--- | :--- | :--- |
| api_type | EB_API_TYPE | str | 否 | 后端平台的类型。支持`"qianfan"`、`"yinian"`和`"aistudio"`，默认是`"qianfan"`。 |
| access_token | EB_ACCESS_TOKEN | str | 否 | 认证鉴权的access token。具体参见[认证鉴权文档](./authentication.md)。 |
| ak | EB_AK | str | 否 | 认证鉴权的API key或access key ID。必须和`sk`同时设置。 |
| sk | EB_SK | str | 否 | 认证鉴权的secret key或secret access key。必须和`ak`同时设置。 |
| max_retries | EB_MAX_RETRIES | int | 否 | 最大请求重试次数。默认值为`0`。 |
| min_retry_delay | EB_MIN_RETRY_DELAY | float | 否 | 请求重试时两次尝试间的最短等待时间，单位为秒。默认值为`1`。 |
| max_retry_delay | EB_MAX_RETRY_DELAY | float | 否 | 请求重试时两次尝试间的最长等待时间（不计随机扰动），单位为秒。默认值为`10`。 |
| proxy | EB_PROXY | str | 否 | 请求使用的代理。 |
