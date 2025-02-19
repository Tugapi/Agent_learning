from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

# 用于在特定位置传入消息列表
chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个人工智能助手。"),
    MessagesPlaceholder("msgs")
])
# 另一种方式
chat_template2 = ChatPromptTemplate.from_messages([
    ("system", "你是一个人工智能助手。"),
    ("placeholder", "{msgs}")
])

if __name__ == '__main__':
    # 传入几条消息就在"msgs"处插入几条消息
    message = chat_template.invoke({"msgs": [HumanMessage(content="你好！"),
                                             HumanMessage(content="你是什么模型？")]})
    print(message)