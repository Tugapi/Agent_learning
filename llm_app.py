# 提示词模版
from langchain_core.prompts import ChatPromptTemplate
# 输出转换
from langchain_core.output_parsers import StrOutputParser
# langchain openai sdk
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是世界级技术专家"),
    ("user", "{input}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

if __name__ == '__main__':
    result = chain.invoke({"input": "写一篇AI agents综述"})
    print(result)
