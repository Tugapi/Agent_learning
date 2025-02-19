import requests
import json

# 同步调用
response = requests.post(
    "http://localhost:8000/openai_ext/invoke",
    json={"input": {"topic": "猫"}}
)
print("同步调用/openai_ext/invoke结果")
print(response.json())

# 流式调用
response = requests.post(
    "http://localhost:8000/openai_ext/stream",
    json={"input": {"topic": "猫"}}
)
print("流式调用/openai_ext/stream结果")
for line in response.iter_lines():
    line = line.decode("utf-8")
    if line.startswith("data: ") and not line.endswith("[DONE]"):
        data = json.loads(line[len("data: "):])
        print(data)
