from langchain_core.prompts import PromptTemplate


prompt_template = PromptTemplate.from_template(
    "讲一个关于{content}的{adjective}笑话。"
)

if __name__ == '__main__':
    result = prompt_template.format(content="猴", adjective="冷")
    print(result)