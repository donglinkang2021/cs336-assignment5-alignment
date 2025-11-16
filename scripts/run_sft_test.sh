set -e  # Exit on error

# first you should run a vllm server with the base model
# e.g., bash run_vllm_server.sh

export WANDB_MODE=offline
export WANDB_DATA_DIR="wandb"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# uv run cs336_alignment/train_sft.py
uv run cs336_alignment/train_sft.py --config-name exp_sft_fa2_omr12k