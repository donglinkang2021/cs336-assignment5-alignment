#!/bin/bash
# Add error handling
set -e  # Exit immediately on error
set -u  # Error on unset variables
set -o pipefail  # Fail a pipeline if any command fails

export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# Use more descriptive variable names and add read-only protection
readonly MODEL_NAME="qwen2.5-math-1.5b-sft"
readonly MODEL_PATH="ckpt/sft_omr12k/best_model"

# Check if the model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path $MODEL_PATH does not exist"
    exit 1
fi

# Run with vLLM backend [Recommended] Fast
echo "Evaluating on Math validation set..."
python evaluate_math.py backend=vllm \
    model.model_name_or_path="$MODEL_PATH" \
    dataset.data_path=data/Math/validation.jsonl \
    output_dir="outputs/${MODEL_NAME}/math_local"

echo "Evaluating on GSM8K test set..."
python evaluate_math.py backend=vllm \
    model.model_name_or_path="$MODEL_PATH" \
    dataset.data_path=data/gsm8k/test.jsonl \
    output_dir="outputs/${MODEL_NAME}/gsm8k_local"

echo "Evaluating on MATH-500 HuggingFace dataset..."
python evaluate_math.py backend=vllm \
    model.model_name_or_path="$MODEL_PATH" \
    dataset.type=huggingface \
    dataset.dataset_name=HuggingFaceH4/MATH-500 \
    dataset.split=test \
    output_dir="outputs/${MODEL_NAME}/math500_hf"

echo "All evaluations completed successfully!"

# Run with Hugging Face backend [So slow]
# python evaluate_math.py backend=hf \
#     model.model_name_or_path="$MODEL_PATH" \
#     dataset.data_path=data/Math/validation.jsonl