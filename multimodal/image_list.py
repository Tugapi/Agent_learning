from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

image_url1 = ""
image_url2 = ""

model = ChatOpenAI(model="gpt-4o")
message = HumanMessage(
    content=[
        {"type": "text", "text": "这两张图片是一个场景的吗？"},
        {"type": "image_url", "image_url": {"url": image_url1}},
        {"type": "image_url", "image_url": {"url": image_url2}},
    ]
)
response = model.invoke([message])
print(response.content)