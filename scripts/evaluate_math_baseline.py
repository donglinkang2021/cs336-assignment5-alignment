"""
Evaluate Qwen 2.5 Math 1.5B zero-shot performance on MATH validation set.

This script:
1. Loads MATH validation examples
2. Formats them using the r1_zero prompt
3. Generates outputs using vLLM
4. Evaluates using the Dr. GRPO reward function
5. Saves results to disk

Usage:
    # Use local dataset
    python scripts/evaluate_math_baseline.py \
        --model-name-or-path models/Qwen2.5-Math-1.5B \
        --dataset-type local \
        --data-path data/Math/validation.jsonl \
        --output-dir outputs/competition_math_results.jsonl \
        --num-gpus 1
    
    # Use HuggingFace dataset
    python scripts/evaluate_math_baseline.py \
        --model-name-or-path models/Qwen2.5-Math-1.5B \
        --dataset-type huggingface \
        --dataset-name HuggingFaceH4/MATH-500 \
        --dataset-split test \
        --output-dir outputs/math500_results.jsonl \
        --num-gpus 1
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from statistics import mean
from typing import Callable, Dict

from datasets import load_dataset, Dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
from xopen import xopen

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

logger = logging.getLogger(__name__)


def load_prompt_template(prompt_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = Path(__file__).parent.parent / "cs336_alignment" / "prompts" / f"{prompt_name}.prompt"
    with open(prompt_path) as f:
        return f.read()

def load_dataset_flexible(
    dataset_type: str,
    data_path: str = None,
    dataset_name: str = None,
    dataset_split: str = None,
) -> Dataset:
    """
    Load dataset from either local file or HuggingFace Hub.
    
    Args:
        dataset_type: Either 'local' or 'huggingface'
        data_path: Path to local jsonl file (for dataset_type='local')
        dataset_name: HuggingFace dataset name (for dataset_type='huggingface')
        dataset_split: Dataset split to load (for dataset_type='huggingface')
    
    Returns:
        HuggingFace Dataset object
    """
    if dataset_type == "local":
        if data_path is None:
            raise ValueError("data_path must be provided when dataset_type='local'")
        logger.info(f"Loading examples from local file: {data_path}...")
        dataset = load_dataset('json', data_files=data_path, split='train')
        logger.info(f"Loaded {len(dataset)} examples")
        return dataset
    elif dataset_type == "huggingface":
        if dataset_name is None or dataset_split is None:
            raise ValueError(
                "dataset_name and dataset_split must be provided when dataset_type='huggingface'"
            )
        logger.info(f"Loading examples from HuggingFace: {dataset_name}, split={dataset_split}...")
        dataset = load_dataset(dataset_name, split=dataset_split)
        logger.info(f"Loaded {len(dataset)} examples")
        return dataset
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}. Must be 'local' or 'huggingface'")


def process_math_example(sample: Dict, prompt_template: str) -> Dict:
    """
    Process a single MATH example by formatting the prompt.
    
    Args:
        sample: A single example from the dataset
        prompt_template: The prompt template to use
    
    Returns:
        Processed sample with 'prompt' and 'ground_truth' fields
    """
    # Handle different field names from different datasets
    question = sample.get("problem", sample.get("question", ""))
    answer = sample.get("solution", sample.get("answer", ""))
    
    # Format the prompt
    sample["prompt"] = prompt_template.format(question=question)
    sample["ground_truth"] = answer
    
    return sample


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    dataset: Dataset,
    eval_sampling_params: SamplingParams,
    output_dir: str,
) -> None:
    """
    Evaluate a language model on a dataset,
    compute evaluation metrics, and serialize results to disk.
    
    Args:
        vllm_model: The vLLM model to evaluate
        reward_fn: The reward function to use for evaluation
        dataset: HuggingFace Dataset with 'prompt' and 'ground_truth' fields
        eval_sampling_params: Sampling parameters for generation
        output_dir: Directory to save results (will contain results.jsonl and summary.jsonl)
    """
    prompts = dataset["prompt"]
    ground_truths = dataset["ground_truth"]
    
    logger.info(f"Generating responses for {len(prompts)} prompts...")
    
    # Generate responses
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    
    # Extract generated texts
    responses = []
    for output in outputs:
        generated_text = output.outputs[0].text
        responses.append(generated_text)
    
    logger.info(f"Evaluating {len(responses)} responses...")
    
    # Evaluate responses
    all_metrics = []
    results = []
    
    for idx, (response, ground_truth) in enumerate(tqdm(
        zip(responses, ground_truths),
        total=len(responses),
        desc="Evaluating"
    )):
        # Compute metrics using reward function
        metrics = reward_fn(response, ground_truth)
        all_metrics.append(metrics)
        
        # Store result - get the original example from dataset
        result = {
            # **dataset[idx],
            "prompt": dataset[idx]["prompt"],
            "ground_truth": dataset[idx]["ground_truth"],
            "response": response,
            "metrics": metrics,
        }
        results.append(result)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results to results.jsonl
    results_path = output_dir / "results.jsonl"
    logger.info(f"Saving detailed results to {results_path}...")
    with xopen(results_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    # Calculate aggregate metrics
    aggregate_metrics = {}
    for key in sorted(all_metrics[0].keys()):
        aggregate_metrics[key] = mean([m[key] for m in all_metrics])
    
    # Calculate category breakdown
    category_1 = sum(1 for m in all_metrics if m["format_reward"] == 1.0 and m["answer_reward"] == 1.0)
    category_2 = sum(1 for m in all_metrics if m["format_reward"] == 1.0 and m["answer_reward"] == 0.0)
    category_3 = sum(1 for m in all_metrics if m["format_reward"] == 0.0 and m["answer_reward"] == 0.0)
    
    total = len(all_metrics)
    
    # Create summary
    summary = {
        "total_examples": total,
        "aggregate_metrics": aggregate_metrics,
        "category_breakdown": {
            "category_1_format1_answer1": {
                "count": category_1,
                "percentage": category_1 / total * 100
            },
            "category_2_format1_answer0": {
                "count": category_2,
                "percentage": category_2 / total * 100
            },
            "category_3_format0_answer0": {
                "count": category_3,
                "percentage": category_3 / total * 100
            }
        }
    }
    
    # Save summary to summary.jsonl
    summary_path = output_dir / "summary.jsonl"
    logger.info(f"Saving summary to {summary_path}...")
    with xopen(summary_path, "w") as f:
        f.write(json.dumps(summary, indent=2) + "\n")
    
    # Log aggregate metrics
    logger.info("=" * 80)
    logger.info("Evaluation Results:")
    logger.info("=" * 80)
    for key, value in aggregate_metrics.items():
        logger.info(f"{key}: {value:.4f}")
    
    # Log category breakdown
    logger.info("=" * 80)
    logger.info("Category Breakdown:")
    logger.info("=" * 80)
    logger.info(f"Category 1 (format=1, answer=1): {category_1} ({category_1/total*100:.2f}%)")
    logger.info(f"Category 2 (format=1, answer=0): {category_2} ({category_2/total*100:.2f}%)")
    logger.info(f"Category 3 (format=0, answer=0): {category_3} ({category_3/total*100:.2f}%)")
    logger.info("=" * 80)


def main(args):
    # Load the model
    logger.info(f"Loading model from {args.model_name_or_path}...")
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.num_gpus,
        trust_remote_code=True,
    )
    
    # Load the prompt template
    logger.info("Loading r1_zero prompt template...")
    prompt_template = load_prompt_template("r1_zero")
    
    # Load the MATH dataset
    dataset = load_dataset_flexible(
        dataset_type=args.dataset_type,
        data_path=args.data_path,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
    )
    
    # Process dataset using map
    logger.info("Processing dataset with prompt template...")
    dataset = dataset.map(
        lambda sample: process_math_example(sample, prompt_template),
        desc="Formatting prompts"
    )
    
    # Print first example for verification
    logger.info("First example:")
    logger.info(json.dumps(dataset[0], indent=2))
    
    # Set up sampling parameters
    # Based on Dr. GRPO: stop when the model completes its answer
    # https://github.com/sail-sg/understand-r1-zero/blob/
    # c18804602b85da9e88b4aeeb6c43e2f08c594fbc/train_zero_math.py#L167
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    
    # Evaluate the model
    evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        dataset=dataset,
        eval_sampling_params=sampling_params,
        output_dir=args.output_dir,
    )
    
    logger.info(f"Evaluation complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    
    parser = argparse.ArgumentParser(description="Evaluate zero-shot MATH baseline")
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="models/Qwen2.5-Math-1.5B",
        help="Path to the model",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["local", "huggingface"],
        default="local",
        help="Type of dataset to load: 'local' for jsonl files or 'huggingface' for HF datasets",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/Math/validation.jsonl",
        help="Path to MATH validation data (for dataset_type='local')",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="HuggingFace dataset name (for dataset_type='huggingface'), e.g., 'HuggingFaceH4/MATH-500'",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default=None,
        help="Dataset split to load (for dataset_type='huggingface'), e.g., 'test'",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/math_baseline",
        help="Directory to save evaluation results (will contain results.jsonl and summary.jsonl)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism",
    )
    
    args = parser.parse_args()
    logger.info("Running: %s", " ".join(sys.argv))
    main(args)
    logger.info("Finished running %s", sys.argv[0])
