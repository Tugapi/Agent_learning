import asyncio

from langchain_core.tools import tool
from pydantic import BaseModel, Field


# 使用tool装饰器创建工具
@tool
def simple_multiply(a: int, b: int) -> int:
    """
    Multiply two integers
    :param a: integer a
    :param b: integer b
    :return: the product of a and b
    """
    return a * b


# 也可创建一个异步实现
@tool
async def amultiply(a: int, b: int) -> int:
    """
    Multiply two integers
    :param a: integer a
    :param b: integer b
    :return: the product of a and b
    """
    return a * b


# 可自定义工具名称，结合pydantic使用检查输入格式，设定工具调用结束后是否直接退出agent loop
class CalculatorInput(BaseModel):
    a: int = Field(description="first integer")
    b: int = Field(description="second integer")


@tool("multiplication-tool", args_schema=CalculatorInput, return_direct=True)
def multiply(a: int, b: int) -> int:
    """
    Multiply two integers
    :param a: integer a
    :param b: integer b
    :return: the product of a and b
    """
    return a * b

print(simple_multiply.name)
print(simple_multiply.description)
print(simple_multiply.args)

print(multiply.name)
print(multiply.description)
print(multiply.args)
print(multiply.invoke({"a": 1, "b": 2}))