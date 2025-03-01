from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

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
# 存储历史消息到缓存
store = {}


def get_session_history(session_id) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# 创建一个带历史会话记录的运行器
chain_with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
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