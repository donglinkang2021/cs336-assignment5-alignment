export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1,2 
uv run -m vllm_sync.server model=models/Qwen2.5-Math-1.5B tensor_parallel_size=2