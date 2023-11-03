import asyncio

from erniebot_agent.chat_models.erniebot import ERNIEBot
from erniebot_agent.message import HumanMessage


async def test_ernie_bot(model="ernie-bot-turbo", stream=False):
    api_type = "aistudio"
    access_token = "<your access token>"
    eb = ERNIEBot(model=model, api_type=api_type, access_token=access_token)
    messages = [
        HumanMessage(content="我在深圳，周末可以去哪里玩？"),
    ]
    res = await eb.async_chat(messages, stream=stream)
    if not stream:
        print(res)
    else:
        async for chunk in res:
            print(chunk)


if __name__ == "__main__":
    asyncio.run(test_ernie_bot(stream=False))
    asyncio.run(test_ernie_bot(stream=True))
