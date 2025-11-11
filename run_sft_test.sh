#!/bin/bash
# Quick test script for SFT training
# This script runs a minimal training job to verify everything works

set -e  # Exit on error

echo "======================================"
echo "SFT Training Quick Test"
echo "======================================"

export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Configuration
MODEL_PATH="models/Qwen2.5-Math-1.5B"
TRAIN_DATA="data/OMR12k/train.jsonl"
VAL_DATA="data/OMR12k/validation.jsonl"
OUTPUT_DIR="ckpt/sft_test"
NUM_EXAMPLES=128
BATCH_SIZE=8
GRAD_ACCUM=2
NUM_EPOCHS=10
EVAL_STEPS=50

echo ""
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Training examples: $NUM_EXAMPLES"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Epochs: $NUM_EPOCHS"
echo ""

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please download the model first."
    exit 1
fi

# Check if data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: Training data not found at $TRAIN_DATA"
    exit 1
fi

if [ ! -f "$VAL_DATA" ]; then
    echo "Error: Validation data not found at $VAL_DATA"
    exit 1
fi

echo "Starting training..."
echo ""

python train_sft.py \
    --model-name-or-path "$MODEL_PATH" \
    --train-data-path "$TRAIN_DATA" \
    --val-data-path "$VAL_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --num-train-examples $NUM_EXAMPLES \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRAD_ACCUM \
    --gradient-checkpointing \
    --learning-rate 5e-6 \
    --num-epochs $NUM_EPOCHS \
    --eval-steps $EVAL_STEPS \
    --save-steps 0 \
    --max-eval-examples 20 \
    --seed 42

echo ""
echo "======================================"
echo "Training completed successfully!"
echo "======================================"
echo ""
echo "To generate samples from the trained model:"
echo "  uv run scripts/generate_samples.py \\"
echo "    --model-name-or-path $OUTPUT_DIR/best_model \\"
echo "    --num-samples 5"
echo ""
echo "To evaluate the full validation set:"
echo "  uv run scripts/evaluate_sft.py \\"
echo "    --model-name-or-path $OUTPUT_DIR/best_model \\"
echo "    --output-dir outputs/eval_test"
echo ""
