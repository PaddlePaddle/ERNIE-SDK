import erniebot

if __name__ == '__main__':
    erniebot.api_type = "qianfan"
    embedding = erniebot.Embedding.create(
        model="ernie-text-embedding",
        input=[
            "我是百度公司开发的人工智能语言模型，我的中文名是文心一言，英文名是ERNIE-Bot，可以协助您完成范围广泛的任务并提供有关各种主题的信息，比如回答问题，提供定义和解释及建议。如果您有任何问题，请随时向我提问。",
            "2018年深圳市各区GDP"
        ])
    print(embedding)
    print(type(embedding))
