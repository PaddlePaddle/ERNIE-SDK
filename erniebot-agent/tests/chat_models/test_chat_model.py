import asyncio
import os

from erniebot_agent.chat_models.erniebot import ERNIEBot
from erniebot_agent.memory import HumanMessage, Message


async def test_ernie_bot(model="ernie-bot-turbo", stream=False):
    api_type = "aistudio"
    access_token = os.getenv("EB_AGENT_ACCESS_TOKEN")  # set your access token as an environment variable
    assert (
        access_token is not None
    ), 'Please set your access token as an environment variable named "EB_AGENT_ACCESS_TOKEN"'
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


async def test_ernie_bot_qianfan(model="ernie-turbo", stream=False, **auth_dict):
    api_type = "qianfan"
    if "access_token" in auth_dict:
        eb = ERNIEBot(model=model, api_type=api_type, access_token=auth_dict["access_token"])
    elif "ak" and "sk" in auth_dict:
        eb = ERNIEBot(model=model, api_type=api_type, ak=auth_dict["ak"], sk=auth_dict["sk"])
    messages = [
        HumanMessage(content="我在深圳，周末可以去哪里玩？"),
    ]
    res = await eb.async_chat(messages, stream=stream)
    assert isinstance(res, Message) and len(res.content) > 0
    if not stream:
        print(res)


if __name__ == "__main__":
    asyncio.run(test_ernie_bot(stream=False))
    asyncio.run(test_ernie_bot(stream=True))
    # asyncio.run(test_ernie_bot_qianfan(access_token=os.getenv("EB_AGENT_ACCESS_TOKEN")))
    asyncio.run(test_ernie_bot_qianfan(ak=os.getenv("EB_AK"), sk=os.getenv("EB_SK")))
