import asyncio
from erniebot_agent.file import GlobalFileManagerHandler

async def fun():
    await GlobalFileManagerHandler().get()

async def main():
    await asyncio.gather(fun(), fun(), fun(), fun())

asyncio.run(main())
