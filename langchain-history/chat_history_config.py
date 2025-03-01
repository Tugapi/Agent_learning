from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec

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


# 为每个用户，每个对话分别存储历史会话记录
def get_session_history(user_id, conversation_id) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = ChatMessageHistory()
    return store[(user_id, conversation_id)]


chain_with_chat_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="用户的唯一标识符。",
            is_shared=True
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="",
            is_shared=True
        )
    ]
)

response = chain_with_chat_history.invoke(
    {"ability": "math", "input": "余弦的定义是什么？"},
    config={"configurable": {"user_id": "123", "conversation_id": "1"}}
)
print(response)

# 包含历史信息再提问
response = chain_with_chat_history.invoke(
    {"ability": "math", "input": "给出具体计算例子"},
    config={"configurable": {"user_id": "123", "conversation_id": "1"}}
)
print(response)

# 新的user_id或conversation_id，不含历史信息
response = chain_with_chat_history.invoke(
    {"ability": "math", "input": "给出具体计算例子"},
    config={"configurable": {"user_id": "123", "conversation_id": "2"}}
)
print(response)