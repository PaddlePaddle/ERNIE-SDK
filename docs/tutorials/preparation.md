# 安装和鉴权

## 安装

### 快速安装

执行如下命令，快速安装最新版本ERNIE Bot Agent（要求Python >= 3.8)。

```shell
# 安装核心模块
pip install --upgrade erniebot-agent

# 安装所有模块
pip install --upgrade erniebot-agent[all]
```

### 源码安装

执行如下命令，使用源码安装ERNIE Bot Agent。

```shell
git clone https://github.com/PaddlePaddle/ERNIE-Bot-SDK.git
cd ERNIE-Bot-SDK

# 切换到特定分支，默认是develop分支
# git checkout -b xxx origin/xxx

# 首先安装Ernie Bot SDK
cd erniebot
pip install .

# 然后安装ERNIE Bot Agent
cd ../erniebot-agent
pip install .           # 安装核心模块
pip install '.[all]'    # 安装所有模块，包括gradio等依赖库
```

如果大家希望二次开发ERNIE Bot Agent，需要额外安装一些依赖库。
```shell
cd ERNIE-Bot-SDK/erniebot-agent
pip install -r dev-requirements.txt
pip install -e '.[all]'
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