import asyncio
from openai import AsyncOpenAI

# DeepSeek R1 API configuration
API_URL = "http://0.0.0.0:8000"

# gets API Key from environment variable OPENAI_API_KEY
client = AsyncOpenAI(
    base_url=f"{API_URL}/v1",
    api_key="EMPTY",
)

async def get_models_id():
    list_completion = await client.models.list()
    return [model.id async for model in list_completion]

async def main() -> None:
    MODEL_NAME = (await get_models_id())[0]
    stream = await client.completions.create(
        model=MODEL_NAME,
        prompt="介绍一下你自己",
        temperature=0.6,
        top_p=0.95,
        max_tokens=32768,
        stream=True,
    )
    async for completion in stream:
        print(completion.choices[0].text, end="")
    print()


asyncio.run(main())