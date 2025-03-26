# 导入tavily检索工具
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

search_tool = TavilySearchResults(max_result=1)
print(search_tool.invoke("今天北京天气怎么样？"))
