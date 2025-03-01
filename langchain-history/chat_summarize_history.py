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

chain_with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history=lambda session_id: temp_chat_history,
    input_messages_key="input",
    history_messages_key="history"
)


def summarize_message(chain_input):
    stored_messages = temp_chat_history.messages
    if len(stored_messages) == 0:
        return False
    summarization_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "user",
                "将上述信息浓缩成一条摘要信息，尽量包含全部细节信息。"
            ),
        ]
    )
    summarization_chain = summarization_prompt | llm
    summary_message = summarization_chain.invoke({"chat_history": stored_messages})
    temp_chat_history.clear()
    temp_chat_history.add_message(summary_message)
    return True


chain_with_summary = RunnablePassthrough(messages_summarize=summarize_message) | chain_with_message_history
response = chain_with_summary.invoke(
    {"input": "我名字是什么？我心情怎么样？我下午在干什么？"},
    config={"configurable": {"session_id": "unused"}}  # 实际不起作用，get_session_history永远返回temp_chat_history
)
print(temp_chat_history.messages)
print(response.content)