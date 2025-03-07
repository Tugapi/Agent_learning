from langchain.output_parsers import YamlOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

model = ChatOpenAI(model="gpt-4o", temperature=0)


# 配合pydantic使用，定义期望的Json输出结构
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
parser = YamlOutputParser(pydantic_object=Joke)
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