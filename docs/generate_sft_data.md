# 生成 Math 的 DeepSeek-R1 SFT 数据

- [generate_competition_math_sft.py](../scripts/generate_competition_math_sft.py) 调用 api 评测

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