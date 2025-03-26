from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 猫的Wikipedia网页
loader = WebBaseLoader("https://zh.wikipedia.org/wiki/%E7%8C%AB")
docs = loader.load()
# 第一个chunk包含第1~第1000个字符，第二个chunk包含第801~第1800个字符
documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
vector = FAISS.from_documents(documents, OpenAIEmbeddings())  # 分块后的网页经embedding处理后存入向量数据库
retriever = vector.as_retriever()

print(retriever.invoke("猫的品种")[0])

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="cat_wiki_search",
    description="搜索猫的Wikipedia网页"
)

