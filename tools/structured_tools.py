import asyncio

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


def simple_multiply(a: int, b: int) -> int:
    """
    Multiply two integers
    :param a: integer a
    :param b: integer b
    :return: the product of a and b
    """
    return a * b


async def amultiply(a: int, b: int) -> int:
    """
    Multiply two integers
    :param a: integer a
    :param b: integer b
    :return: the product of a and b
    """
    return a * b


class CalculatorInput(BaseModel):
    a: int = Field(description="first integer")
    b: int = Field(description="second integer")


async def main():
    # StructuredTool.from_function类方法比@tool具备更多可配置性，一个工具同时支持同步和异步调用
    calculator = StructuredTool.from_function(
        func=simple_multiply,  # 指定同步函数，在同步上下文中调用工具时使用
        coroutine=amultiply,  # 指定异步函数，在异步上下文中调用工具时使用
        name="multiplication-tool",
        description="multiply integers",
        args_schema=CalculatorInput,
        return_direct=True
    )
    # 同步调用
    print(calculator.invoke({"a": 2, "b": 3}))
    # 异步调用
    print(await calculator.ainvoke({"a": 2, "b": 5}))
    print(calculator.name)
    print(calculator.description)


asyncio.run(main())
