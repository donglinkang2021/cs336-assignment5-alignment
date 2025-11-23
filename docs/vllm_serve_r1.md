# 使用 vllm 部署 R1

> Reference: https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3.html

R1 的文件已经下载在：

```bash
/inspire/hdd/global_public/public_models/deepseek-ai/DeepSeek-R1
```

- vllm 要开 enable-prefix-caching 和 enable-chunked-prefill
- 如果开 gpu-memory-utilization=0.95 会一直报错
- 不要用 blackwell 的环境变量，就只是使用下面配置就好（环境 H200x8）

```bash
export VLLM_USE_DEEP_GEMM=1 
export VLLM_ALL2ALL_BACKEND="deepep_high_throughput" # or "deepep_low_latency"
uv run vllm serve /inspire/hdd/global_public/public_models/deepseek-ai/DeepSeek-R1 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization=0.90 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-model-len=65536 \
  --enable-expert-parallel
```

## Usage

```bash
curl -X POST http://0.0.0.0:8000/v1/chat/completions \
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