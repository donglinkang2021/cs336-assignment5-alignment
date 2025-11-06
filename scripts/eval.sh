# Use local dataset
export HF_DATASETS_OFFLINE=1
python scripts/evaluate_math_baseline.py \
    --model-name-or-path models/Qwen2.5-Math-1.5B \
    --dataset-type local \
    --data-path data/Math/validation.jsonl \
    --output-dir outputs/qwen2.5-math-1.5b/math_local \
    --num-gpus 1

python scripts/evaluate_math_baseline.py \
    --model-name-or-path models/Qwen2.5-Math-1.5B \
    --dataset-type local \
    --data-path data/gsm8k/test.jsonl \
    --output-dir outputs/qwen2.5-math-1.5b/gsm8k_local \
    --num-gpus 1

# Use HuggingFace dataset
python scripts/evaluate_math_baseline.py \
    --model-name-or-path models/Qwen2.5-Math-1.5B \
    --dataset-type huggingface \
    --dataset-name HuggingFaceH4/MATH-500 \
    --dataset-split test \
    --output-dir outputs/qwen2.5-math-1.5b/math500_hf \
    --num-gpus 1

# Use local dataset
python scripts/evaluate_math_baseline.py \
    --model-name-or-path models/Qwen2.5-Math-1.5B-Instruct \
    --dataset-type local \
    --data-path data/Math/validation.jsonl \
    --output-dir outputs/qwen2.5-math-1.5b-inst/math_local \
    --num-gpus 1

python scripts/evaluate_math_baseline.py \
    --model-name-or-path models/Qwen2.5-Math-1.5B-Instruct \
    --dataset-type local \
    --data-path data/gsm8k/test.jsonl \
    --output-dir outputs/qwen2.5-math-1.5b-inst/gsm8k_local \
    --num-gpus 1

# Use HuggingFace dataset
python scripts/evaluate_math_baseline.py \
    --model-name-or-path models/Qwen2.5-Math-1.5B-Instruct \
    --dataset-type huggingface \
    --dataset-name HuggingFaceH4/MATH-500 \
    --dataset-split test \
    --output-dir outputs/qwen2.5-math-1.5b-inst/math500_hf \
    --num-gpus 1