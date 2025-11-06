# Use local dataset
python scripts/evaluate_math_baseline.py \
    --model-name-or-path models/Qwen2.5-Math-1.5B \
    --dataset-type local \
    --data-path data/Math/validation.jsonl \
    --output-path outputs/math_local_results.jsonl \
    --num-gpus 1

python scripts/evaluate_math_baseline.py \
    --model-name-or-path models/Qwen2.5-Math-1.5B \
    --dataset-type local \
    --data-path data/gsm8k/test.jsonl \
    --output-path outputs/gsm8k_local_results.jsonl \
    --num-gpus 1

# Use HuggingFace dataset
python scripts/evaluate_math_baseline.py \
    --model-name-or-path models/Qwen2.5-Math-1.5B \
    --dataset-type huggingface \
    --dataset-name HuggingFaceH4/MATH-500 \
    --dataset-split test \
    --output-path outputs/math500_hf_results.jsonl \
    --num-gpus 1