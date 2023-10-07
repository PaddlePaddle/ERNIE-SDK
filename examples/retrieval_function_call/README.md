# 检索增强和函数调用演示应用

## 简介

我们构建了一个名为**ERNIE Bot 城市建设法规标准小助手**的演示应用，展示了如何借助检索增强和函数调用的力量拓展大型模型的专有领域知识。我们使用[PaddleNLP Pipelines流水线系统](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/pipelines)和 [ERNIE Bot 语义向量Embedding](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/alj562vvu)搭建了本地语义检索服务，为大型模型提供了检索本地知识库的能力。此外，借助ERNIE Bot SDK的[函数调用Function Call](https://github.com/PaddlePaddle/ERNIE-Bot-SDK#%E5%87%BD%E6%95%B0%E8%B0%83%E7%94%A8function-calling)能力，我们可以根据上下文和用户提出的具体问题，让大型模型在回答问题时决定是否采用检索增强方式，或是直接回答。这一设计在赋予了大模型更多领域知识的同时，也保留了领域知识以外的通用大模型对话能力。

![retrieval_function_call_demo](https://github.com/PaddlePaddle/ERNIE-Bot-SDK/assets/11987277/ad7edf01-620a-427d-81f1-368501b764ad)

## 安装依赖

首先，请确保Python版本>=3.8。然后执行如下命令：

```shell
cd examples/retrieval_function_call
pip install -r requirements.txt
```

## 下载数据

我们已经预备好了来自[中华人民共和国住房和城乡建设部规章库](https://www.mohurd.gov.cn/dynamic/rule/library)的部分规章文件，可以通过以下的命令下载和解压：

```shell
wget https://paddlenlp.bj.bcebos.com/datasets/examples/construction_regulations.tar
tar xvf construction_regulations.tar
```

## 启动服务

可以通过以下的命令启动本地演示服务:

```shell
python demo.py
    --api_key <your/api/key> \
    --secret_key <your/secret/key> \
    --file_paths  construction_regulations \
    --port 8081
```

参数含义说明
- `api_key`: ERNIE Bot SDK的api key
- `secret_key`: ERNIE Bot SDK的secret key
- `file_paths`: 用于构建检索索引的文档库路径。默认为'construction_regulations'
- `index_name`: 构建的检索索引名称。如本地语义检索索引已经存在，则直接读取已经构建好的索引。默认为'construct_demo_index'
- `retriever_top_k`: 检索的召回数量，默认为5
- `chunk_size`: 构建检索的文本切片大小，默认为384个tokens
- `host`: Gradio Demo的host name, 默认为'localhost'
- `port`: Gradio Demo的port, 默认为8081
