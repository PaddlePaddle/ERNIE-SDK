import erniebot

if __name__ == "__main__":
    # Set AK/SK for erniebot.Image.create
    # erniebot.ak = 'xx'
    # erniebot.sk = 'xx'
    erniebot.api_type = "yinian"
    response = erniebot.Image.create(
        model='ernie-vilg-v2',
        prompt='请帮我画一个开心的袋熊',
        width=512,
        height=512,
        version='v2',
        image_num=1)
    print(response)
