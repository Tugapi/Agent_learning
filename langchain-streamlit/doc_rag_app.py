import streamlit as st
import tempfile
import os
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import TextLoader
from langchain_core.tools import create_retriever_tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler


# 设置streamlit页面标题和布局
st.set_page_config(page_title="文档问答", layout="wide")
# 设置标题
st.title("文档问答")
# 上传txt文件，允许上传多个
uploaded_files = st.sidebar.file_uploader(
    label="上传txt文档", type=["txt"], accept_multiple_files=True
)
if not uploaded_files:
    st.info("请先上传txt文档")
    st.stop()


@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    # 读取上传文件，存入本地目录
    docs = []
    temp_dir = tempfile.TemporaryDirectory(dir=r"D:\\")
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        # 使用TextLoader加载文档
        loader = TextLoader(temp_filepath, encoding="utf-8")
        docs.extend(loader.load())

    # 文档分割
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 获取文档向量表示，存入向量数据库
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(splits, embeddings)

    retriever = vectordb.as_retriever()
    return retriever


retriever = configure_retriever(uploaded_files)
# 若session_state中没有历史消息记录或用户清空了历史，进行消息记录初始化
if "messages" not in st.session_state or st.sidebar.button("清空聊天记录"):
    st.session_state["messages"] = [{"role": "assistant", "content": "您好，我是文档问答助手。"}]
# 加载历史消息记录
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="doc_rag",
    description="在用户上传的文档中检索用户问题相关内容"
)
tools = [retriever_tool]
# 创建聊天消息历史记录
msgs = StreamlitChatMessageHistory()
# 创建对话缓冲区内存
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_memory", output_key="output"
)
# 指令模板，强制大模型调用rag工具
instructions = """
你是一个查询文档来回答问题的AI智能体。你可以使用文档检索工具，基于检索内容回答问题。
你可能不查询文档就知道答案，但你仍然要先查询文档获取答案。
如果你从文档中找不到任何信息用于回答问题，只需返回"这个问题我没有在文档中找到答案。"。
"""
#  基础提示词模板
base_prompt_template = """
{instruction}

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

‍'''
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: {input}
Observation: the result of the action
‍'''

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

‍'''
Thought: Do I need to use a tool? No
Final Answer: [your response here]
‍'''

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
"""

prompt_template = PromptTemplate.from_template(base_prompt_template)
# 填充部分提示词模版
prompt = prompt_template.partial(instructions=instructions)

llm = ChatOpenAI()
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors="没有从文档中检索到相关内容"
)
# 聊天输入框
user_query = st.chat_input(placeholder="请提问")
if user_query:
    # 用户消息存入session_state
    st.session_state.messages.append({"role": "user", "content": user_query})
    # 显示用户消息
    st.chat_message("user").write(user_query)
    with st.chat_message("assistant"):
        # 创建streamlit回调处理器，agent执行过程日志显示在streamlit container
        st_cb = StreamlitCallbackHandler(st.container())
        config = {"callbacks": st_cb}
        response = agent_executor.invoke({"input": user_query}, config)
        st.session_state.messages.append({"role": "assistant", "content": response["output"]})
        # 显示agent响应输出
        st.write(response["output"])