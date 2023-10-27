# 模型总览

ERNIE Bot SDK支持的所有模型如下：

| 模型名称 | 说明 | 功能 | 支持该模型的后端 | 输入token数量上限 |
|:--- | :--- | :--- | :--- | :--- |
| ernie-bot | 文心一言模型。具备优秀的知识增强和内容生成能力，在文本创作、问答、推理和代码生成等方面表现出色。 | 对话补全，函数调用 | qianfan，aistudio | 3000 |
| ernie-bot-turbo | 文心一言模型。相比erniebot模型具备更快的响应速度和学习能力，API调用成本更低。 | 对话补全 | qianfan，aistudio | 3000 |
| ernie-bot-4 | 文心一言模型。基于文心大模型4.0版本的文心一言，具备目前文心一言系列模型中最优的理解和生成能力。 | 对话补全，函数调用 | qianfan，aistudio | 3000 |
| ernie-bot-8k | 文心一言模型。在ernie-bot模型的基础上增强了对长对话上下文的支持，输入token数量上限为7000。 | 对话补全，函数调用 | qianfan，aistudio | 7000 |
| ernie-text-embedding | 文心百中语义模型。支持计算最多384个token的文本的向量表示。 | 语义向量 | qianfan，aistudio | 384*16 |
| ernie-vilg-v2 | 文心一格模型。 | 文生图 | yinian | 200 |
