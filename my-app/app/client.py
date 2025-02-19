from langchain.schema.runnable import RunnableMap
from langchain_core.prompts import ChatPromptTemplate
from langserve import RemoteRunnable

openai = RemoteRunnable("http://localhost:8000/openai")
prompt = ChatPromptTemplate.from_messages(
    [("system", "你是一个喜欢写故事的助手"), ("user", "写一个主题是{topic}的故事")]
)

chain = prompt | RunnableMap({
    "openai": openai
})
print("同步调用/openai/invoke结果")
response = chain.invoke({"topic": "鹦鹉"})
print(response)

openai_str_parser = RemoteRunnable("http://localhost:8000/openai_str_parser")
chain_str_parser = prompt | RunnableMap({
    "openai" : openai_str_parser
})
print("StrOutputParser结果")
str_parser_response = chain_str_parser.invoke({"topic": "鹦鹉"})
print(str_parser_response)

print("流式调用/openai/stream结果")
for chunk in chain.stream({"topic": "鹦鹉"}):
    print(chunk, end="", flush=True)
    print(chunk["openai"], end="", flush=True)