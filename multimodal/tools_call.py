from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool


@tool
def weather_tool(weather: Literal["晴朗的", "多云的", "多雨的", "下雪的"]):
    print("天气为：", weather)


model = ChatOpenAI(model="gpt-4o")
model_with_tools = model.bind_tools([weather_tool])
image_url1 = ""
image_url2 = ""

message = HumanMessage(
    content=[
        {"type": "text", "text": "用中文描述两张图片中的天气。"},
        {"type": "image_url", "image_url": {"url": image_url1}},
        {"type": "image_url", "image_url": {"url": image_url2}},
    ]
)
response = model.invoke([message])
print(response.tool_calls)