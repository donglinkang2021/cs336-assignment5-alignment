#!/bin/bash
# Add error handling
set -e  # Exit immediately on error
set -u  # Error on unset variables
set -o pipefail  # Fail a pipeline if any command fails

export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Define models to test
MODELS=(
    "Qwen/Qwen2.5-Math-1.5B"
    "Qwen/Qwen2.5-Math-1.5B-Instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    "Qwen/Qwen2.5-Math-7B"
    "Qwen/Qwen2.5-Math-7B-Instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)

# Define temperature and top_p combinations to test
# Format: "temperature:top_p"
COMBINATIONS=(
    "0.0:1.0"
    "0.6:0.95"
)

echo "Starting grid search evaluation..."
echo "Total models to test: ${#MODELS[@]}"
echo "Total combinations per model: ${#COMBINATIONS[@]}"
echo "Total evaluations: $((${#MODELS[@]} * ${#COMBINATIONS[@]}))"
echo "-------------------------------------------"

# Loop through each model
for MODEL_PATH in "${MODELS[@]}"; do
    echo ""
    echo "###################################################"
    echo "Testing model: ${MODEL_PATH}"
    echo "###################################################"
    
    # Check if it's a DeepSeek model
    if [[ "$MODEL_PATH" == *"DeepSeek"* ]]; then
        MAX_TOKENS_ARG="generation.max_tokens=32768"
    else
        MAX_TOKENS_ARG=""
    fi
    
    # Loop through each combination
    for combo in "${COMBINATIONS[@]}"; do
        # Split the combination into temperature and top_p
        IFS=':' read -r temp topp <<< "$combo"
        
        # Extract model name for save path
        MODEL_NAME=$(basename "$MODEL_PATH")
        SAVE_PATH="${MODEL_NAME}-temperature${temp}-topp${topp}"
        
        echo ""
        echo "==================================================="
        echo "Testing: ${MODEL_PATH}"
        echo "Parameters: temperature=${temp}, top_p=${topp}"
        echo "==================================================="
        
        # Run evaluation with conditional max_tokens
        if [ -n "$MAX_TOKENS_ARG" ]; then
            uv run cs336_alignment/evaluate_math.py num_gpus=4 \
                model.model_name_or_path="$MODEL_PATH" \
                generation.temperature="$temp" \
                generation.top_p="$topp" \
                "$MAX_TOKENS_ARG" \
                datasets=[competition_math_500,competition_math_5k,gsm8k,math500,amc,aime,aime22,aime23,aime24,aime25] \
                output_dir="eval_results2/${SAVE_PATH}/"
        else
            uv run cs336_alignment/evaluate_math.py num_gpus=4 \
                model.model_name_or_path="$MODEL_PATH" \
                generation.temperature="$temp" \
                generation.top_p="$topp" \
                datasets=[competition_math_500,competition_math_5k,gsm8k,math500,amc,aime,aime22,aime23,aime24,aime25] \
                output_dir="eval_results2/${SAVE_PATH}/"
        fi
        
        echo "Completed: ${MODEL_PATH} with temperature=${temp}, top_p=${topp}"
    done
done

echo ""
echo "==================================================="
echo "All evaluations completed successfully!"
echo "==================================================="