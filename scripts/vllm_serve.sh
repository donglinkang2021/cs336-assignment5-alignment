export VLLM_USE_DEEP_GEMM=1 
export VLLM_ALL2ALL_BACKEND="deepep_high_throughput" # or "deepep_low_latency"
uv run vllm serve /inspire/hdd/global_public/public_models/deepseek-ai/DeepSeek-R1 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization=0.90 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-model-len=16384 \
  --enable-expert-parallel