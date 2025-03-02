from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# 按url方式传入，大模型访问url获取图片
image_url = ""

model = ChatOpenAI(model="gpt-4o")
message = HumanMessage(
    content=[
        {"type": "text", "text": "用中文描述这张图片。"},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
)
response = model.invoke([message])
print(response.content)