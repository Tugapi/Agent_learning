from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI(
    title="LangChain 服务器",
    version="1.0",
    description="使用LangChain Runnable接口的简单API服务器"
)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
# add_routes(app, NotImplemented)

add_routes(
    app,
    ChatOpenAI(model="gpt-3.5-turbo"),
    path="/openai"
)

parser = StrOutputParser()
add_routes(
    app,
    ChatOpenAI(model="gpt-3.5-turbo") | parser,
    path="/openai_str_parser"
)

prompt = ChatPromptTemplate.from_template("告诉我一个关于{topic}的笑话")
add_routes(
    app,
    prompt | ChatOpenAI(model="gpt-3.5-turbo"),
    path="/openai_ext"
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
