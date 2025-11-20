# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is CS336 Spring 2025 Assignment 5: Alignment, focusing on training and evaluating language models with SFT (Supervised Fine-Tuning), DPO (Direct Preference Optimization), and GRPO (Group Relative Policy Optimization) methods. The project uses Qwen 2.5 Math 1.5B as the base model for mathematical reasoning tasks.

## Commands

### Setup & Testing

Install dependencies:
```bash
uv sync --no-install-package flash-attn
uv sync
# or: uv add flash-attn
```

Run all tests:
```bash
uv run pytest
```

Run specific test suites:
```bash
uv run pytest -k test_sft
uv run pytest -k test_dpo
uv run pytest -k test_grpo
```

Tests require implementing functions in `tests/adapters.py`. Initially all tests fail with `NotImplementedError`.

### SFT Training

Launch VLLM server (must run first):
```bash
bash scripts/run_vllm_server.sh
```

Run SFT training:
```bash
# Using shell script
bash scripts/run_sft_test.sh

# Direct execution with specific config
uv run cs336_alignment/train_sft.py --config-name exp_sft_omr4096_12k
```

The training script:
- Uses Qwen2.5-Math-1.5B with flash attention
- Trains on OMR12k (OpenMathReasoning) dataset
- Performs validation every 50 steps on multiple datasets
- Logs to Weights & Biases
- Uses memory-efficient AdamW optimizer with CPU-offloaded states

### Evaluation

Run evaluation with vLLM backend (recommended):
```bash
bash scripts/eval.sh
```

Custom evaluation:
```bash
# vLLM backend (fast)
uv run cs336_alignment/evaluate_math.py \
    model.model_name_or_path="Qwen/Qwen2.5-Math-1.5B" \
    generation.temperature=0.6 \
    generation.top_p=0.95 \
    datasets=[competition_math,gsm8k,math500] \
    output_dir="eval_results/test/"

# HuggingFace backend
uv run cs336_alignment/evaluate_math.py \
    model.model_name_or_path="models/Qwen2.5-Math-1.5B" \
    backend="hf"
```

Sync W&B logs:
```bash
bash scripts/sync_wandb.sh
```

### Visualization

Launch visualization server for evaluation results:
```bash
bash scripts/launch_server.sh
```

## Architecture

### Core Modules (`cs336_alignment/`)

**SFT Training (`train_sft.py`)**
- Main training loop with checkpointing
- Uses Hydra for configuration management
- Integrates VLLM for evaluation during training
- Memory-efficient training with gradient checkpointing

**SFT Utilities (`sft_utils.py`)**
- `tokenize_prompt_and_output()`: Creates response masks
- `get_response_log_probs()`: Extracts log probabilities
- `compute_entropy()`: Memory-efficient entropy calculation with chunking
- `selective_log_softmax()`: Efficient log-softmax
- `sft_microbatch_train_step()`: Microbatch training with gradient accumulation

**Math Evaluation (`evaluate_math.py`)**
- Supports vLLM and HuggingFace backends
- Computes format and answer rewards with detailed metrics
- Outputs JSONL + summary JSON

**Math Grading (`drgrpo_grader.py`)**
- `r1_zero_reward_fn()`: Main reward function (4-class: format × answer)
- Multiple grading strategies: string matching, sympy, math_verify
- Answer extraction from LaTeX boxed notation

**Optimizer (`optimizer.py`)**
- `MemoryEfficientAdamW`: Custom AdamW with CPU-offloaded states
- Uses pinned memory for efficient GPU-CPU transfers
- Significantly reduces GPU memory usage

**Utils (`utils.py`)**
- Prompt template loading
- Utility functions for model training

### Test Adapters (`tests/adapters.py`)

This is the primary implementation file. All tests call adapter functions that students must implement:

**Implemented (complete in `sft_utils.py`):**
- `run_tokenize_prompt_and_output()`
- `run_compute_entropy()`
- `run_get_response_log_probs()`
- `run_sft_microbatch_train_step()`
- `run_masked_normalize()`

**To implement:**
- `run_compute_group_normalized_rewards()`: GRPO reward normalization
- `run_compute_naive_policy_gradient_loss()`: Policy gradient loss
- `run_compute_grpo_clip_loss()`: GRPO clip loss
- `run_compute_policy_gradient_loss()`: Policy gradient wrapper
- `run_masked_mean()`: Masked mean computation
- `run_grpo_microbatch_train_step()`: GRPO training step

Optional RLHF/safety functions:
- `get_packed_sft_dataset()`: Dataset packing
- `run_iterate_batches()`: Batch iteration
- `run_parse_mmlu_response()`: MMLU answer parsing
- `run_parse_gsm8k_response()`: GSM8K answer parsing
- `run_compute_per_instance_dpo_loss()`: DPO loss

### Configuration (`conf/`)

Uses Hydra for hierarchical configuration:

**Base configs:**
- `sft_config.yaml`: Main SFT training config
- `eval_math.yaml`: Evaluation config
- `vllm_server.yaml`: VLLM server config

**Experiment configs:**
- `exp_sft_omr12k.yaml`: OMR12k full training
- `exp_sft_omr2048_12k.yaml`: Filtered to 2048 tokens
- `exp_sft_omr4096_12k.yaml`: Filtered to 4096 tokens

**Validation datasets (`val_datasets/`):**
- 9 math datasets: competition_math, gsm8k, math500, etc.

### Scripts (`scripts/`)

**Training & Evaluation:**
- `run_vllm_server.sh`: Start VLLM inference server
- `run_sft_test.sh`: Launch SFT training
- `eval.sh`, `eval_models.sh`: Run evaluation

**Utilities:**
- `sync_wandb.sh`: Sync Weights & Biases logs
- `launch_server.sh`: Visualize evaluation results
- `visualize_results.py`: Visualization utilities
- `save_eval_datasets.py`: Dataset preparation
- `batch_size_check.py`: Memory profiling

### Data Structure

**Training data (`data/`):**
- `OMR12k/`: OpenMathReasoning 12k (main training set)
- `OMR12k-2048/`, `OMR12k-4096/`: Length-filtered versions

**Evaluation data (`data/eval/`):**
- Multiple math benchmarks: GSM8K, competition math, AIME, MATH, etc.

**Results:**
- `eval_results/`: Evaluation outputs with various temperature/top-p settings

## Key Concepts

### Reward System

Math problems use 4-class rewards (format × answer):
- **00**: Wrong format, wrong answer
- **01**: Wrong format, correct answer
- **10**: Correct format, wrong answer
- **11**: Correct format, correct answer

Grading checks:
1. Format: Answer is boxed in LaTeX (`\boxed{}`)
2. Answer: Answer correctness (string match, sympy, math_verify)

### Memory Efficiency

The training system emphasizes memory efficiency:
- Flash attention 2 for efficient attention computation
- Gradient checkpointing
- CPU-offloaded optimizer states (pinned memory)
- Chunked operations for large tensors
- Microbatch training steps

### Evaluation Architecture

Supports two backends:
- **vLLM** (production): Fast inference with continuous batching
- **HuggingFace** (development): Standard transformers

Both produce identical output format.

### GRPO Training

Group Relative Policy Optimization:
- Multiple rollouts per prompt (group_size parameter)
- Group-based reward normalization
- Multiple loss variants: no_baseline, reinforce_with_baseline, grpo_clip

## Implementation Workflow

1. **Complete `tests/adapters.py`**: Implement missing adapter functions
2. **Run tests**: Verify implementation with `uv run pytest`
3. **Train model**: Run SFT training script
4. **Evaluate**: Assess performance on validation sets
5. **Iterate**: Refine implementation based on results

## Dependencies

Key libraries:
- `transformers>=4.50.0`: Model loading and training
- `flash-attn>=2.8.3`: Efficient attention
- `vllm>=0.11.0`: Fast inference
- `math-verify>=0.8.0`: Answer verification
- `alpaca-eval`: Evaluation framework
- `wandb>=0.19.8`: Experiment tracking
- `hydra-core>=1.3.0`: Configuration management
- `accelerate>=1.5.2`: Distributed training

Python requirement: 3.11 or 3.12 (3.13 not supported)