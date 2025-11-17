#!/bin/bash
# Add error handling
set -e  # Exit immediately on error
set -u  # Error on unset variables
set -o pipefail  # Fail a pipeline if any command fails

export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=3

# Use more descriptive variable names and add read-only protection
readonly MODEL_PATH="Qwen/Qwen2.5-Math-1.5B"
readonly SAVA_PATH="Qwen/Qwen2.5-Math-1.5B-temperature0.6-topp0.95"

# Run with vLLM backend [Recommended] Fast
echo "Evaluating..."
uv run cs336_alignment/evaluate_math.py \
    model.model_name_or_path="$MODEL_PATH" \
    generation.temperature=0.6 \
    generation.top_p=0.95 \
    datasets=[competition_math,gsm8k,math500] \
    output_dir="eval_results/${SAVA_PATH}/"

echo "All evaluations completed successfully!"
