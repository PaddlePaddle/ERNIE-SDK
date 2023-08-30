# 参数配置

ERNIE Bot SDK参数配置，主要涉及认证鉴权、后端平台等信息。

ERNIE Bot SDK支持两种参数配置的方法：1）使用环境变量，2）使用Python API。

1）使用环境变量：
``` {.copy}
export EB_API_TYPE="<EB-API-TYPE>"
```

2）使用Python API：
``` {.py .copy}
import erniebot
erniebot.api_type = "<EB-API-TYPE>"
```

注意：使用Python API设置的优先级高于使用环境变量设置。

ERNIE Bot SDK支持的参数，具体介绍如下：

| API参数名称   | 环境变量名称  |  类型   |  必须设置 |  描述   |
| :---         | :----       | :----  | :---- |  :---- |
| api_type     | EB_API_TYPE | string | 否 | 设置后端平台的类型，支持`'qianfan'`和`'yinian'`，默认是`'qianfan'`。|
| ak           | EB_AK       | string | 否 | 设置认证鉴权的access key。必须和sk同时设置。 |
| sk           | EB_SK       | string | 否 | 设置认证鉴权的secret key。必须和ak同时设置。 |
| access_token | EB_ACCESS_TOKEN | string | 否 | 设置认证鉴权的access token，推荐大家优先使用ak和sk。如果设置了access token，优先使用该access token。如果access token没有设置或者失效，并且设置了ak和sk，会自动使用ak和sk获取access token。|
| access_token_path | EB_ACCESS_TOKEN_PATH | string | 否 | 设置存有access token的文件路径，推荐大家优先使用ak和sk。`access_token_path`生效原理和access token相同。|
| proxy        | EB_PROXY    | string | 否 | 设置请求的代理 。|
| timeout      | EB_TIMEOUT  | float  | 否 | 设置请求超时的时间。如果设置了timeout，请求失败后会再次请求，直到成功或者超过设置的时间。|
