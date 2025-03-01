from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

# 预先给定的chat history
temp_chat_history = ChatMessageHistory()
temp_chat_history.add_user_message("我叫Tim，你好。")
temp_chat_history.add_ai_message("你好。")
temp_chat_history.add_user_message("我今天心情很开心。")
temp_chat_history.add_user_message("我下午在打篮球。")
print(temp_chat_history.messages)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个乐于助人的助手。尽力回答所有问题。提供的聊天历史包含与你交谈的用户的真实信息。"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

llm = ChatOpenAI(model="gpt-4")
chain = prompt | llm


def trim_messages(chain_input):
    stored_messages = temp_chat_history.messages
    if len(stored_messages) <= 2:
        return False
    temp_chat_history.clear()
    for message in stored_messages[-2:]:
        temp_chat_history.add_message(message)
    return True


chain_with_chat_history = RunnableWithMessageHistory(
    chain,
    get_session_history=lambda session_id: temp_chat_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)
chain_with_trimming = RunnablePassthrough.assign(messages_trimmed=trim_messages) | chain_with_chat_history

# 历史中包含这条信息
response = chain_with_trimming.invoke(
    {"input": "我下午在干什么？"},
    config={"configurable": {"session_id": "unused"}}  # 实际不起作用，get_session_history永远返回temp_chat_history
)
print(response)


response = chain_with_trimming.invoke(
    {"input": "我叫什么名字？"},
    config = {"configurable": {"session_id": "unused"}}  # 实际不起作用，get_session_history永远返回temp_chat_history
)
print(response)