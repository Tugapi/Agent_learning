from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're an assistant who's good at {ability}. Respond in 20 words or fewer."
        ),
        # 历史消息列表
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]
)
model = ChatOpenAI(model="gpt-4")
chain = prompt | model

REDIS_URL = "redis://localhost:6379/0"


def get_message_history(session_id) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(session_id, url=REDIS_URL)


chain_with_message_history = RunnableWithMessageHistory(
    chain,
    get_message_history,
    input_messages_key="input",
    history_messages_key="history",
)

response = chain_with_message_history.invoke(
    {"ability": "math", "input": "余弦的定义是什么？"},
    config={"configurable": {"session_id": "abc123"}}
)
print(response)

# 包含历史信息再提问
response = chain_with_message_history.invoke(
    {"ability": "math", "input": "给出具体计算例子"},
    config={"configurable": {"session_id": "abc123"}}
)
print(response)

# 新的session_id，不含历史信息
response = chain_with_message_history.invoke(
    {"ability": "math", "input": "给出具体计算例子"},
    config={"configurable": {"session_id": "def123"}}
)
print(response)
