# ERNIEBot-Agent QA Bot

ERNIEBot-Agent QA Bot是一个EB-Agent使用教学机器人，基于`FunctionAgentWithRetrieval`，该工具旨在辅助用户解决与EB-Agent相关的问题，帮助用户更快的使用`erniebot_agent`库，搭建属于自己的Agent。


## 架构

此应用基于`FunctionAgentWithRetrieval`（后续`RetrievalAgent`上线后将同步更换），将此仓库中相关模块的markdown文件以及ipynb的示例代码文件向量化并通过自定义检索工具检索，实现EB-Agent教学机器人。

### 自定义检索工具

此应用中的检索工具基于`langchain`的`faiss`本地向量库，同时基于此应用特性，用户可能需要了解具体的代码实现。因此在实现时同时检索召回说明文档的内容(存储于db)以及相关的代码内容(存储于module_code_db)。

```python
class FaissSearch:
    def __init__(self, db, embeddings, module_code_db):
        self.db = db
        self.module_code_db = module_code_db
        self.embeddings = embeddings
```
## 如何开始

**注意：** 建库的过程比较缓慢，请耐心等待。

> 第一步：下载项目源代码，请确保您已经安装了erniebot_agent以及erniebot
```bash
git clone https://github.com/PaddlePaddle/ERNIE-Bot-SDK.git
cd ernie-agent/applications/eb-agent-qa-bot
pip install ernie_agent
```
> 第二步：如果是第一次运行，请先初始化向量库
```bash
python question_bot.py --init=True --access-token <aistudio-access-token>
```
> 如果已经初始化过向量库，直接运行即可
```bash
python question_bot.py --access-token <aistudio-access-token>
```