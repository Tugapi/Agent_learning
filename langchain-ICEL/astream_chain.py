import asyncio
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "给我讲一个关于{topic}的笑话。"
)
model = ChatOpenAI()
parser = StrOutputParser()

chain = prompt | model | parser


async def async_stream():
    async for chunk in chain.astream({"topic": "鹦鹉"}):
        print(chunk, end="|", flush=True)


if __name__ == '__main__':
    asyncio.run(async_stream())
