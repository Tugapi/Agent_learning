from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

examples = [
    {
        "question": "谁的寿命更长，穆罕默德·阿里还是艾伦·图灵？",
        "answer":
            """
            这里需要跟进问题吗：是的。
            跟进：穆罕默德·阿里去世时多大？
            中间答案：穆罕默德·阿里去世时74岁。
            跟进：艾伦·图灵去世时多大？
            中间答案：艾伦·图灵去世时41岁。
            所以最终答案是：穆罕默德·阿里。
            """
    },
    {
        "question": "《大白鲨》和《皇家赌场》的导演来自同一个国家吗？",
        "answer":
            """
            这里需要跟进问题吗：是的。
            跟进：《大白鲨》的导演是谁？
            中间答案：《大白鲨》的导演是Steven Spielberg。
            跟进：Steven Spielberg来自哪里？
            中间答案：美国。
            跟进：《皇家赌场》的导演是谁？
            中间答案：《皇家赌场》的导演是Martin Campbell。
            跟进：Martin Campbell来自哪里？
            中间答案：新西兰。
            所以最终答案是：穆罕默德·阿里。
            """
    }
]

# 不再把所有examples插入提示词，而是先计算用户输入和examples的embeddings的相似度，只插入相似的example
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    # 用于存储embeddings，执行搜索的向量库
    Chroma,
    # 筛选出的事例数
    k=1
)

example_prompt = PromptTemplate(input_variables=["question", "answer"], template="问题：{question}\n{answer}")

prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix="问题：{input}",
    input_variables=["input"]
)

if __name__ == '__main__':
    question = "谁的寿命更长，乔治·华盛顿还是亚伯拉罕·林肯？"
    selected_examples = example_selector.select_examples({"question": question})
    print("选出的事例：\n")
    for example in selected_examples:
        for k, v in example.items():
            print(f"{k}: {v}")
            print("\n")

    print(prompt.format(input="谁的寿命更长，乔治·华盛顿还是亚伯拉罕·林肯？"))