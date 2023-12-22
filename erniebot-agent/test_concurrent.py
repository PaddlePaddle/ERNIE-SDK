import asyncio
from erniebot_agent.file import GlobalFileManager

async def fun():
    await GlobalFileManager().get()

async def main():
    await asyncio.gather(fun(), fun(), fun(), fun())

asyncio.run(main())
