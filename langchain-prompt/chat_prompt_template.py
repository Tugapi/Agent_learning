from langchain_core.prompts import ChatPromptTemplate

'''
    通过消息数组创建聊天信息模版
    数组每个消息元组元素代表一条消息，元组第一个元素为消息角色，第二个元素为消息内容
'''

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个人工智能助手，你的名字是{name}。"),
        ("human", "你好！"),
        ("ai", "我很好，谢谢！"),
        ("human", "{user_input}")
    ]
)

if __name__ == '__main__':
    message = chat_template.format_messages(name="Bob", user_input="你叫什么名字？")
    print(message)