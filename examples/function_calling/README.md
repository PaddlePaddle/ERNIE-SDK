# 函数调用功能演示应用

函数调用功能演示应用支持基础的对话补全功能，同时，模型可能在对话中要求发起函数调用，这些函数可以是应用预先提供的，也可以是用户自定义的。该应用可供大家体验ERNIE Bot SDK的函数调用功能，也可以用于调试本地代码、以进一步优化函数调用效果。

## 安装依赖

首先，请确保Python版本>=3.8。然后执行如下命令：

```shell
cd examples/function_calling
pip install -r requirements.txt
```

## 运行应用

执行如下命令：

```shell
python function_calling_demo.py --port 8188
```

然后，使用本地浏览器打开[http://localhost:8188](http://localhost:8188)。

### 界面介绍

### 典型使用流程

使用前，请参考[认证鉴权文档](../../docs/authentication.md#%E7%94%B3%E8%AF%B7%E5%8D%83%E5%B8%86%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%B9%B3%E5%8F%B0%E7%9A%84aksk)申请千帆平台的AK/SK用于服务调用鉴权。

### 添加自定义函数
