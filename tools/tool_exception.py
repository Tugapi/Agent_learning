from langchain_core.tools import StructuredTool
from langchain_core.tools import ToolException


def get_weather(city: str) -> int:
    """
    获取指定城市的天气。
    """
    raise ToolException(f"不存在{city}的天气信息。")


get_weather_tool = StructuredTool.from_function(
    func=get_weather,
    handle_tool_error=True  # 若设置为False，抛出错误；若设置为True，则tool会处理错误，返回错误文本，不抛出错误
)

response = get_weather_tool.invoke({"city": "unknown city"})
print(response)

get_weather_tool2 = StructuredTool.from_function(
    func=get_weather,
    handle_tool_error="没找到该城市。"  # 也可设置为字符串，则会覆盖原错误文本，不抛出错误
)

response2 = get_weather_tool2.invoke({"city": "unknown city"})
print(response2)


def handle_error(e: ToolException) -> str:
    return f"get_weather工具执行时发生如下错误：{e.args[0]}"


get_weather_tool3 = StructuredTool.from_function(
    func=get_weather,
    handle_tool_error=handle_error  # 也可设置为处理错误的函数
)

response3 = get_weather_tool3.invoke({"city": "unknown city"})
print(response3)