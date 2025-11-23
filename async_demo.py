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
