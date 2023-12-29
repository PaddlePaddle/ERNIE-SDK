# 安装和鉴权

## 安装

### 源码安装

执行如下命令，使用源码安装ERNIE Bot Agent（要求Python >= 3.8)。

```shell
git clone https://github.com/PaddlePaddle/ERNIE-Bot-SDK.git
cd ERNIE-Bot-SDK

# 切换到特定分支，默认是develop分支
# git checkout -b xxx origin/xxx

# 首先安装Ernie Bot
pip install ./erniebot

# 然后安装ERNIE Bot Agent
pip install ./erniebot-agent            # 安装核心模块
#pip install './erniebot-agent/.[all]'   # 也可以加上[all]一次性安装所有模块，包括gradio等依赖库
```

如果大家希望二次开发ERNIE Bot Agent，需要额外安装一些依赖库。

```shell
pip install -r erniebot-agent/dev-requirements.txt
pip install -e './erniebot-agent/.[all]'
```

### 快速安装（暂不支持）

执行如下命令，快速安装最新版本ERNIE Bot Agent（要求Python >= 3.8)。

```shell
# 安装核心模块
pip install --upgrade erniebot-agent

# 安装所有模块
pip install --upgrade erniebot-agent[all]
```


## 鉴权

大家在使用ERNIE Bot Agent之前，需要完成鉴权步骤：

* 在[AI Studio星河社区](https://aistudio.baidu.com/index)注册并登录账号
* 在个人中心的[访问令牌页面](https://aistudio.baidu.com/index/accessToken)获取用户凭证`Access Token`
* 通过环境变量或者`Python`代码设置`Access Token`

```shell
export EB_AGENT_ACCESS_TOKEN="<your access token>"
```

```python
import os
os.environ["EB_AGENT_ACCESS_TOKEN"] = "<your access token>"
```