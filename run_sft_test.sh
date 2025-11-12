set -e  # Exit on error

# first you should run a vllm server with the base model
# e.g., bash run_vllm_server.sh

export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

uv run train_sft.py
