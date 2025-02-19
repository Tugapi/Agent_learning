import asyncio
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI()
parser = JsonOutputParser()

chain = model | parser


async def async_stream():
    async for chunk in chain.astream(
            "以JSON格式输出西班牙、法国、日本的国家名及其人口列表"
            "使用一个带有'countries'外部键的字典，其中包含国家列表"
            "每个国家应有键'name'和'population'"
    ):
        print(chunk, flush=True)


if __name__ == '__main__':
    asyncio.run(async_stream())
