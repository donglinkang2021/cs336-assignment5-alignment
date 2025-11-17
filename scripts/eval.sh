#!/bin/bash
# Add error handling
set -e  # Exit immediately on error
set -u  # Error on unset variables
set -o pipefail  # Fail a pipeline if any command fails

export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=3

# Use more descriptive variable names and add read-only protection
# readonly MODEL_NAME="qwen2.5-math-1.5b-original"
# readonly MODEL_PATH="models/Qwen2.5-Math-1.5B"
readonly MODEL_NAME="qwen2.5-math-1.5b-sft-omr2048-12k/best_model"
readonly MODEL_PATH="ckpt/exp_sft_omr2048_12k/best_model"

# Check if the model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path $MODEL_PATH does not exist"
    exit 1
fi

# Run with vLLM backend [Recommended] Fast
echo "Evaluating..."
uv run cs336_alignment/evaluate_math.py \
    model.model_name_or_path="$MODEL_PATH" \
    datasets=[competition_math,gsm8k,math500] \
    output_dir="results_eval/${MODEL_NAME}/"

echo "All evaluations completed successfully!"
