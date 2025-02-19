import asyncio
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4")


async def async_stream():
    events = []
    async for event in model.astream_events("hello", version="v2"):
        events.append(event)
    print(events)


if __name__ == '__main__':
    asyncio.run(async_stream())