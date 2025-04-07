from langchain_community.tools import TavilySearchResults
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START
from pydantic import BaseModel, Field
import asyncio
import operator
from typing import TypedDict, List, Annotated, Tuple, Union, Literal
from dotenv import load_dotenv

load_dotenv()

tools = [TavilySearchResults(max_results=1)]

prompt = hub.pull("wfh/react-agent-executor")
prompt.pretty_print()

model = ChatOpenAI(model="gpt-4o", temperature=0)
# 获得React CompiledGraph对象
agent_executor = create_react_agent(model, tools, messages_modifier=prompt)
# 测试图的搜索工具调用能力
agent_executor.invoke({"messages": [("user", "现在上海的天气怎么样？",)]})


# 定义PlanExecutor类，存储输入、规划、历史步骤和历史响应
class PlanExecutor(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


# 定义Plan结构类，描述未来要执行的计划
class Plan(BaseModel):
    steps: List[str] = Field(description="需要执行的各个步骤，按照顺序排列")


planner_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """对于给定的目标，提出一个简单的逐步计划。这个计划应该包含独立的任务，如果正确执行将得出正确的答案。不要添加任何多余的步骤。最后一步的结果应该是最终答案。确保每一步都有所有必要的信息 - 不要跳过步骤。"""),
        ("placeholder", "{messages}")
    ]
)
planner = planner_prompt | model.with_structured_output(Plan)


# 定义Response结构类，描述对用户的回答
class Response(BaseModel):
    response: str


# 定义行为结构类，描述下一步操作
class Act(BaseModel):
    action: Union[Plan, Response] = Field(
        description="要执行的行为。如果要回应用户，使用Response。如果要进一步使用工具获取答案，使用Plan。"
    )


replanner_prompt = ChatPromptTemplate.from_template(
    """对于给定的目标，提出一个简单的逐步计划。这个计划应该包含独立的任务，如果正确执行将得出正确的答案。不要添加任何多余的步骤。最后一步的结果应该是最终答案。确保每一步都有所有必要的信息 - 不要跳过步骤。
        你的目标是：
        {input}
        
        你的原计划是：
        {plan}
        
        你目前已完成的步骤是：
        {past_steps}
        相应地更新你的计划。如果不需要更多步骤并且可以返回给用户回答，那么就这样回应。如果需要，填写计划。只添加还需完成的步骤，不要把已完成的步骤作为计划的一部分。
        """
)
replanner = replanner_prompt | model.with_structured_output(Act)


async def main():
    # 获取步骤列表
    async def plan_step(state: PlanExecutor):
        plan = await planner.ainvoke({"messages": [("user", state["input"],)]})
        return {"plan": plan.steps}

    # 执行步骤
    async def execute_step(state: PlanExecutor):
        plan = state["plan"]
        plan_str = '\n'.join(f"{i+1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        task_formatted = f"对于以下步骤：{plan_str}\n\n你的任务是执行第{1}步，{task}"
        agent_response = await agent_executor.ainvoke({"messages": [("user", task_formatted)]})
        return {"past_steps": state["past_steps"] + [task, agent_response["messages"][-1].content]}

    # 重新执行步骤
    async def replan_step(state: PlanExecutor):
        output = await replanner.ainvoke(state)
        if isinstance(output.action, Response):
            return {"response": output.action.response}
        else:
            return {"plan": output.action.steps}

    # 判断是否结束
    def should_end(state: PlanExecutor) -> Literal["agent", "__end__"]:
        if "response" in state:
            return "__end__"
        else:
            return "agent"

    workflow = StateGraph(PlanExecutor)
    workflow.add_node("planner", plan_step)
    workflow.add_node("agent", execute_step)
    workflow.add_node("replan", replan_step)
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "agent")
    workflow.add_edge("agent", "replan")
    workflow.add_conditional_edges("replan", should_end)

    app = workflow.compile()
    config = {"recursion_limit": 50}  # 递归限制为50
    inputs = {"input": "2024年奥运会男子100米自由泳冠军的国籍是什么？请用中文回答。"}
    # 异步执行图
    async for event in app.astream(inputs, config=config):
        for key, value in event.items():
            if key != "__end__":
                print(value)

if __name__ == '__main__':
    asyncio.run(main())