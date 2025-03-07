from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

model = ChatOpenAI(model="gpt-4o", temperature=0)


class Joke(BaseModel):
    """
    {
      "setup": "...",
      "punchline": "..."
    }
    """
    setup: str = Field(description="设置笑话的问题")
    punchline: str = Field(description="解决笑话的答案")


joke_query = "告诉我一个关于鹦鹉的笑话。"
parser = JsonOutputParser(pydantic_object=Joke)
# 在没有pydantic的情况下使用JsonOutputParser，则输出结构随机的Json
# parser = JsonOutputParser()
# OutputParser原理是把结构要求转化成提示词给模型
prompt = PromptTemplate(
    template="回答用户的查询。\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
print("format_instructions:")
print(parser.get_format_instructions())

chain = prompt | model | parser
response = chain.invoke({"query": joke_query})
print(response)
# 相比于PydanticOutputParser，JsonOutputParser支持格式化流式调用输出
for s in chain.stream({"query": joke_query}):
    print(s)