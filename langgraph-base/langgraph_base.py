from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
# 用于持久化状态
from langgraph.checkpoint.memory import MemorySaver
# 状态图及状态
from langgraph.graph import END, StateGraph, MessagesState
# 工具节点
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv


load_dotenv()


@tool
def search(query: str):
    """
    模拟一个搜索天气API
    :param query: 想查询天气的城市
    :return: 城市的天气状况
    """
    if "上海" in query.lower() or "shanghai" in query.lower():
        return "现在30℃，有雾。"
    return "现在35℃，晴天。"


tools = [search]
# 创建工具节点
tool_node = ToolNode(tools)
model = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)


# 路由函数，控制是否继续执行
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    # 如果模型调用了工具，则转到"tools"节点
    if last_message.tool_calls:
        return "tools"
    # 否则停止回复用户
    else:
        return END


# 调用模型函数
def call_model(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")  # 设置入口节点
workflow.add_conditional_edges(source="agent", path=should_continue)  # 添加条件边
workflow.add_edge("tools", "agent")  # 添加普通边
# 初始化内存以在图运行时存储持久化状态
checkpointer = MemorySaver()
# 编译图成为CompiledGraph对象，其实现了Runnable接口，可像其他Langchain可执行对象一样使用
app = workflow.compile(checkpointer=checkpointer)
# 执行图
final_state = app.invoke(
    {"messages": [HumanMessage(content="上海的天气怎么样？")]},
    config={"configurable": {"thread_id": 42}}
)
result = final_state["messages"][-1].content
print(result)
final_state = app.invoke(
    {"messages": [HumanMessage(content="我刚才问的哪个城市的天气？")]},
    config={"configurable": {"thread_id": 42}}
)
result = final_state["messages"][-1].content
print(result)
# 可视化图
graph_png = app.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(graph_png)