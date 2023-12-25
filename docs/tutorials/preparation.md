# 准备

## 安装

### pip快速安装

执行如下命令，快速安装最新版本ERNIE Bot Agent（要求Python >= 3.8)。

```shell
pip install --upgrade erniebot-agent
```

### 源码安装

```shell
git clone https://github.com/PaddlePaddle/ERNIE-Bot-SDK.git

# 安装Ernie Bot SDK
cd ERNIE-Bot-SDK/erniebot
pip install .

# 安装ERNIE Bot Agent
cd ../erniebot-agent
pip install .
```

## 鉴权

大家在使用ERNIE Bot Agent之前，需要完成鉴权步骤：

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