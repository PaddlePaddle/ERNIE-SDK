# 简介
ERNIE Bot Agent是文心一言Agent框架，旨在助力大家快速开发大模型应用。

# 快速上手

## 安装
执行如下命令，快速安装最新版本ERNIE Bot Agent（要求Python >= 3.8)。

```shell
pip install --upgrade erniebot-agent
```

## 鉴权
大家在使用ERNIE Bot Agent之前，需要进行鉴权步骤：

* 在[AI Studio星河社区](https://aistudio.baidu.com/index)注册并登录账号
* 在个人中心的[访问令牌页面](https://aistudio.baidu.com/index/accessToken)获取用户凭证`Access Token`
* 通过环境变量或者`Python`代码设置`Access Token`

```shell
export EB_AGENT_ACCESS_TOKEN="your access token"
```

```python
import os
os.environ["EB_AGENT_ACCESS_TOKEN"] = "your access token"
```

## 智能体Agent

创建并使用Agent