import base64
import httpx
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# 按base64方式传入，先获取图片二进制数据到本地，转码为base64编码后传给大模型
image_url = ""
image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

model = ChatOpenAI(model="gpt-4o")
message = HumanMessage(
    content=[
        {"type": "text", "text": "用中文描述这张图片。"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
    ]
)
response = model.invoke([message])
print(response.content)