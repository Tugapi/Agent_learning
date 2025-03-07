# pip install defusedxml
"""
建议使用XMLOutputParser前安装defusedxml库
XMLOutputParser默认使用defusedxml作为解析器，以防止潜在的XML漏洞
如果未安装defusedxml，XMLOutputParser将回退使用Python标准库xml解析XML数据
"""
from langchain_core.output_parsers import XMLOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o", temperature=0)
movie_query = "生成Steven Spielberg的电影作品列表，按日期降序排列"
# 支持选择期望包含的tags，（以prompt形式传给模型，层级关系由模型判断，没有对输出的硬约束，可能输出错误格式）
parser = XMLOutputParser(tags=["Movies", "Title" "Year"])
prompt = PromptTemplate(
    template="回答用户的查询。\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
print("format_instructions:")
print(parser.get_format_instructions())

chain = prompt | model | parser
response = chain.invoke({"query": movie_query})
print(response)
# 支持流式调用
for s in chain.stream({"query": movie_query}):
    print(s)