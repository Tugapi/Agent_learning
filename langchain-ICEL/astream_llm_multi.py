import asyncio
from langchain_openai import ChatOpenAI


async def task1():
    gpt4 = ChatOpenAI(model="gpt-4")
    chunks = []
    async for chunk in gpt4.astream("天空是什么颜色？"):
        chunks.append(chunk)
        if len(chunks) == 2:
            print(chunks[1])
        print(chunk.content, end="|", flush=True)


async def task2():
    gpt4o = ChatOpenAI(model="gpt-4o")
    chunks = []
    async for chunk in gpt4o.astream("讲个关于鹦鹉的笑话。"):
        chunks.append(chunk)
        if len(chunks) == 2:
            print(chunks[1])
        print(chunk.content, end="|", flush=True)

async def main():
    # 同步调用
    await task1()
    await task2()
    # 异步调用
    # await asyncio.gather(task1(), task2())

asyncio.run(main())