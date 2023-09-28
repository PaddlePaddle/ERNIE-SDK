import erniebot

if __name__ == '__main__':
    erniebot.api_type = "qianfan"
    response = erniebot.ChatFile.create(
        messages=[{
            "role": "user",
            "content": "你好"
        }, {
            "role": "assistant",
            "content": "你好，我是文心一言"
        }, {
            "role": "user",
            "content":
            "<file>AI_Frameworks.pdf</file><url>https://ernie-bot-sdk.bj.bcebos.com/AI_Frameworks.pdf</url>"
        }, {
            "role": "assistant",
            "content":
            "以下是该文档的简要内容：AI框架技术不断发展，已形成较为完整的体系，包括基础层、组件层和生态层。全球AI框架繁荣发展，多元化竞合态势逐渐显现，赋能学术科研和产业应用。为了应对未来多样化挑战，AI框架有六大技术趋势。虽然AI框架生态远未成熟，但未来发展空间广阔。"
        }, {
            "role": "user",
            "content": "应对未来的技术挑战，AI框架有哪些独特的技术特色？"
        }],
        stream=False)
    print(response.get_result())
