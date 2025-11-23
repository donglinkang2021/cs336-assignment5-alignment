# 生成 Math 的 DeepSeek-R1 SFT 数据

- [generate_competition_math_sft.py](../scripts/generate_competition_math_sft.py) 调用 api 评测 Math 12500 条数据并记录

```bash
curl -X POST http://ds-r1-0528-671b-64k-ep.api.sii.edu.cn/v1/chat/completions \
    -H "Accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{
    "model": "deepseek-r1-0528-ep",
        "messages": [{
        "role": "user",
        "content": "你是谁"
        }],
        "max_tokens": 1000,
        "presence_penalty": 1.03,
        "frequency_penalty": 1.0,
        "seed": null,
        "temperature": 0.6,
        "top_p": 0.95,
        "stream": false
    }'
```

```python
from openai import OpenAI

# DeepSeek R1 API configuration
API_URL = "http://ds-r1-0528-671b-64k-ep.api.sii.edu.cn"
MODEL_NAME = "deepseek-r1-0528-ep"

client = OpenAI(
    base_url=f"{API_URL}/v1",
    api_key="EMPTY",
)

def test_chat_stream():
    """Test streaming chat completion
    
    DeepSeek-R1 Usage Recommendations:
    - Temperature: 0.5-0.7 (0.6 recommended) to prevent endless repetitions
    - No system prompt: all instructions should be in user prompt
    - For math problems: include "Please reason step by step, and put your final answer within \\boxed{}."
    - Enforce thinking pattern by starting response with "<think>\\n"
    """
    template = "{question} Please reason step by step, and put your final answer within \\boxed{{}}."
    question = "Reverse the string 'deepseek-reasoner'."
    # question = "How many 'r's in the word 'strawberry'? Let's think step by step."
    print(template.format(question=question))
    chat_completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": template.format(question=question)}, # renosaer-keespeed
        ],
        max_tokens=8192,
        temperature=0.6,  # Recommended: 0.5-0.7 range
        top_p=0.95,
        extra_body={
            'repetition_penalty': 1.05,
        },
        stream=True,
        # Enforce thinking pattern
        extra_headers={
            'X-Enforce-Thinking': 'true'
        } if False else {}  # Set to True to enforce "<think>\n" prefix
    )
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end='', flush=True)
    print()

if __name__ == '__main__':
    print("\n--- Testing DeepSeek R1 Stream ---")
    test_chat_stream()

# python openai_demo.py
```

```python
import asyncio
import httpx
import os
import time

headers = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}

API_URL = os.environ.get("API_URL", "http://ds-r1-0528-671b-64k-ep.api.sii.edu.cn/v1/chat/completions")
MODEL_NAME = os.environ.get("MODEL_NAME", "deepseek-r1-0528-ep")
TIMEOUT = 300.0  # seconds

async def ask_one(question: str, client: httpx.AsyncClient) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]
    }
    try:
        resp = await client.post(API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {e}"

async def batch_ask_openai(questions: list[str], max_concurrency: int = 20):
    start = time.time()
    semaphore = asyncio.Semaphore(max_concurrency)

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:

        async def sem_task(q):
            async with semaphore:
                return await ask_one(q, client)

        tasks = [sem_task(q) for q in questions]
        responses = await asyncio.gather(*tasks)

    duration = time.time() - start
    print(f"\nSent {len(questions)} requests in {duration:.2f} seconds.")
    return responses

# 示例：100个请求
# 实测结果：Sent 100 requests in 17.19 seconds.
if __name__ == "__main__":
    q_base = "第 {} 个问题：介绍一下你自己"
    question_list = [q_base.format(i) for i in range(1, 101)]

    results = asyncio.run(batch_ask_openai(question_list, max_concurrency=20))

    for i in range(100):
        print(f"\n[Q{i+1}] {question_list[i]}\n[A{i+1}] {results[i]}")

# python async_demo.py
```