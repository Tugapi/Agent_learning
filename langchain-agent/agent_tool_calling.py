from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor

from tools_tavily import search_tool
from tools_retriever import retriever_tool

model = ChatOpenAI(model="gpt-4")
tools = [search_tool, retriever_tool]
prompt = hub.pull("hwchase17/openai-functions-agent")  # agent用提示词模版
print(prompt.messages)  # 额外有一个variable_name='agent_scratchpad'的MessagesPlaceholder
"""
[
    SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], input_types={}, partial_variables={}, template='You are a helpful assistant'), additional_kwargs={}), 
    MessagesPlaceholder(variable_name='chat_history', optional=True), 
    HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], input_types={}, partial_variables={}, template='{input}'), additional_kwargs={}), 
    MessagesPlaceholder(variable_name='agent_scratchpad')
]
"""
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

print(agent_executor.invoke({"input": "宠物猫常见品种有什么？"}))
print(agent_executor.invoke({"input": "今天上海天气怎么样？"}))

#  可添加记忆功能
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
store = {}


def get_session_history(user_id, conversation_id) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = ChatMessageHistory()
    return store[(user_id, conversation_id)]


agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="用户的唯一标识符。",
            is_shared=True
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="",
            is_shared=True
        )
    ]
)

response = agent_with_chat_history.invoke(
    {"input": "宠物猫常见品种有什么？"},
    config={"configurable": {"user_id": "abc123", "conversation_id": "1"}}
)
print(response)


response_with_chat_history = agent_with_chat_history.invoke(
{"input": "以上品种中那种体型最大？"},
    config={"configurable": {"user_id": "abc123", "conversation_id": "1"}}
)