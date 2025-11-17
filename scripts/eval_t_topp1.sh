#!/bin/bash
# Add error handling
set -e  # Exit immediately on error
set -u  # Error on unset variables
set -o pipefail  # Fail a pipeline if any command fails

export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=2

# Use more descriptive variable names and add read-only protection
readonly MODEL_PATH="Qwen/Qwen2.5-Math-1.5B-Instruct"

# Define temperature and top_p combinations to test
# Format: "temperature:top_p"
COMBINATIONS=(
    "0.0:1.0"
    "0.6:0.95"
    "0.6:1.0"
    "1.0:0.95"
    "1.0:1.0"
)

echo "Starting grid search evaluation..."
echo "Total combinations to test: ${#COMBINATIONS[@]}"
echo "-------------------------------------------"

# Loop through each combination
for combo in "${COMBINATIONS[@]}"; do
    # Split the combination into temperature and top_p
    IFS=':' read -r temp topp <<< "$combo"
    
    SAVE_PATH="Qwen/Qwen2.5-Math-1.5B-Instruct-temperature${temp}-topp${topp}"
    
    echo ""
    echo "==================================================="
    echo "Testing: temperature=${temp}, top_p=${topp}"
    echo "==================================================="
    
    # Run evaluation
    uv run cs336_alignment/evaluate_math.py \
        model.model_name_or_path="$MODEL_PATH" \
        generation.temperature="$temp" \
        generation.top_p="$topp" \
        datasets=[competition_math,gsm8k,math500] \
        output_dir="eval_results1/${SAVE_PATH}/"
    
    echo "Completed: temperature=${temp}, top_p=${topp}"
done

echo ""
echo "==================================================="
echo "All evaluations completed successfully!"
echo "==================================================="