"""
Unified evaluation script for MATH dataset with selectable inference backend (vLLM or HF).

This script:
1. Loads MATH validation examples.
2. Formats them using a specified prompt template.
3. Generates outputs using either vLLM or Hugging Face Transformers.
4. Evaluates using a reward function (e.g., Dr. GRPO's).
5. Saves detailed and summary results to disk.
6. Uses Hydra for configuration management.
"""

import re
import json
import logging
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, List

import hydra
import torch
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from xopen import xopen

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.config.eval_config import ScriptArguments
from cs336_alignment.utils import load_prompt_template

# Conditional import for vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

logger = logging.getLogger(__name__)

def sample_dataset(dataset: Dataset, num_samples:int=None) -> Dataset:
    if num_samples:
        num_samples = min(num_samples, len(dataset))
        logger.info(f"Sampling {num_samples} examples from the dataset.")
        indices = random.sample(range(len(dataset)), num_samples)
        dataset = dataset.select(indices)
    return dataset


def process_math_example(sample: Dict, prompt_template: str) -> Dict:
    """
    Process a single MATH example by formatting the prompt.
    """
    question = sample.get("problem", sample.get("question", ""))
    answer = sample.get("solution", sample.get("expected_answer", sample.get("answer", "")))
    
    sample["prompt"] = prompt_template.format(question=question)
    sample["ground_truth"] = answer
    
    return sample


def evaluate_and_save_results(
    prompts: List[str],
    responses: List[str],
    ground_truths: List[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    output_dir: str,
) -> None:
    """
    Evaluate responses, compute metrics, and serialize results to disk.
    """
    logger.info(f"Evaluating {len(responses)} responses...")
    
    all_metrics = []
    results = []
    
    for prompt, response, ground_truth in tqdm(
        zip(prompts, responses, ground_truths),
        total=len(responses),
        desc="Evaluating"
    ):
        metrics = reward_fn(response, ground_truth, False)
        all_metrics.append(metrics)
        
        result = {
            "prompt": prompt,
            "ground_truth": ground_truth,
            "response": response,
            "metrics": metrics,
        }
        results.append(result)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / "results.jsonl"
    logger.info(f"Saving detailed results to {results_file}...")
    with xopen(results_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    aggregate_metrics = {key: mean([m[key] for m in all_metrics]) for key in sorted(all_metrics[0].keys())}
    
    total = len(all_metrics)
    category_1 = sum(1 for m in all_metrics if m["format_reward"] == 1.0 and m["answer_reward"] == 1.0)
    category_2 = sum(1 for m in all_metrics if m["format_reward"] == 1.0 and m["answer_reward"] == 0.0)
    category_3 = sum(1 for m in all_metrics if m["format_reward"] == 0.0 and m["answer_reward"] == 0.0)
    category_4 = sum(1 for m in all_metrics if m["format_reward"] == 0.0 and m["answer_reward"] == 1.0)
    
    summary = {
        "total_examples": total,
        "aggregate_metrics": aggregate_metrics,
        "category_breakdown": {
            "category_1_format1_answer1": {"count": category_1, "percentage": category_1 / total * 100 if total > 0 else 0},
            "category_2_format1_answer0": {"count": category_2, "percentage": category_2 / total * 100 if total > 0 else 0},
            "category_3_format0_answer0": {"count": category_3, "percentage": category_3 / total * 100 if total > 0 else 0},
            "category_4_format0_answer1": {"count": category_4, "percentage": category_4 / total * 100 if total > 0 else 0},
        }
    }
    
    summary_file = output_path / "summary.json"
    logger.info(f"Saving summary to {summary_file}...")
    with xopen(summary_file, "w") as f:
        f.write(json.dumps(summary, indent=2) + "\n")

    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Summary")
    logger.info("=" * 80)
    for key, value in aggregate_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    logger.info("-" * 80)
    logger.info("Category Breakdown:")
    logger.info(f"  Correct (Format & Answer): {category_1} ({summary['category_breakdown']['category_1_format1_answer1']['percentage']:.2f}%)")
    logger.info(f"  Correct Format, Wrong Answer: {category_2} ({summary['category_breakdown']['category_2_format1_answer0']['percentage']:.2f}%)")
    logger.info(f"  Wrong Format, Wrong Answer: {category_3} ({summary['category_breakdown']['category_3_format0_answer0']['percentage']:.2f}%)")
    logger.info(f"  Wrong Format, Correct Answer: {category_4} ({summary['category_breakdown']['category_4_format0_answer1']['percentage']:.2f}%)")
    logger.info("=" * 80)


def run_vllm_evaluation(cfg: ScriptArguments, datasets: Dict[str, Dataset]):
    """Run evaluation using the vLLM backend."""
    if not VLLM_AVAILABLE:
        raise ImportError("vLLM is not installed. Please install it to use the 'vllm' backend.")

    logger.info(f"Loading model with vLLM from {cfg.model.model_name_or_path}...")
    llm = LLM(
        model=cfg.model.model_name_or_path,
        dtype=cfg.model.dtype,
        tensor_parallel_size=cfg.num_gpus,
        trust_remote_code=True,
    )
    
    for dataset_name, dataset in datasets.items():
        logger.info(f"Evaluating dataset: {dataset_name} with {len(dataset)} examples...")
        # Convert generation config to dict for SamplingParams
        gen_dict = OmegaConf.to_container(cfg.generation, resolve=True)
        sampling_params = SamplingParams(**gen_dict)
        
        prompts = dataset["prompt"]
        logger.info(f"Generating responses for {len(prompts)} prompts with vLLM...")
        outputs = llm.generate(prompts, sampling_params)
        responses = [output.outputs[0].text for output in outputs]
        
        evaluate_and_save_results(
            prompts=prompts,
            responses=responses,
            ground_truths=dataset["ground_truth"],
            reward_fn=r1_zero_reward_fn,
            output_dir=f"{cfg.output_dir}/{dataset_name}",
        )


def run_hf_evaluation(cfg: ScriptArguments, datasets: Dict[str, Dataset]):
    """Run evaluation using the Hugging Face Transformers backend."""
    logger.info(f"Loading model with Transformers from {cfg.model.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name_or_path,
        dtype=cfg.model.dtype,
        device_map="auto",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model.eval()

    for dataset_name, dataset in datasets.items():
        logger.info(f"Evaluating dataset: {dataset_name} with {len(dataset)} examples...")

        prompts = [prompt for prompt in dataset["prompt"]]
        
        # Convert generation config to dict for generate()
        generation_kwargs = {
            'max_new_tokens': cfg.generation.max_tokens,
            'do_sample': True,
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            'temperature': cfg.generation.temperature,
            'top_p': cfg.generation.top_p,
        }
        generation_config = GenerationConfig(**generation_kwargs)

        logger.info(f"Generating responses for {len(prompts)} prompts with HF Transformers...")

        # Batch generation
        batch_size = 1
        all_prompts_text = []
        all_completions_text = []
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating batches"):
            batch_prompts = prompts[i:i + batch_size]
            
            prompt_inputs = tokenizer(
                batch_prompts, return_tensors="pt",
                padding=True, padding_side="left",
                add_special_tokens=True,
            ).to(model.device)
            
            with torch.no_grad():
                prompt_completion_ids = model.generate(
                    **prompt_inputs,
                    generation_config=generation_config,
                )
            
            prompt_length = prompt_inputs['input_ids'].size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            
            batch_prompts_text = tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            batch_prompts_text = [
                re.sub(rf"^({re.escape(tokenizer.pad_token)})+", "", text) for text in batch_prompts_text
            ]
            
            batch_completions_text = tokenizer.batch_decode(
                completion_ids, skip_special_tokens=True
            )
            
            all_prompts_text.extend(batch_prompts_text)
            all_completions_text.extend(batch_completions_text)

        evaluate_and_save_results(
            prompts=all_prompts_text,
            responses=all_completions_text,
            ground_truths=dataset["ground_truth"],
            reward_fn=r1_zero_reward_fn,
            output_dir=f"{cfg.output_dir}/{dataset_name}",
        )


@hydra.main(config_path="../conf", config_name="eval_math", version_base=None)
def main(cfg: ScriptArguments):
    # Disable core dump generation to avoid creating core.* files
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    
    # Set random seed for reproducibility
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    logger.info("Starting evaluation script...")
    # Use OmegaConf.to_yaml to handle the dataclass structure
    logger.info(f"Running with config:\n{OmegaConf.to_yaml(cfg)}")
    
    prompt_template = load_prompt_template(cfg.prompt_name)
    
    datasets = {}
    for dataset_name, dataset_cfg in cfg.val_datasets.items():
        logger.info(f"Processing dataset: {dataset_name}")
        dataset = load_dataset(**dataset_cfg)
        dataset = dataset.map(
            lambda sample: process_math_example(sample, prompt_template),
            desc="Formatting prompts"
        )
        datasets[dataset_name] = sample_dataset(dataset, cfg.num_samples)
        
    logger.info("First processed example:")
    for key, value in datasets.items():
        logger.info(f"Dataset: {key}, Number of examples: {len(value)}")
    
    if cfg.backend == "vllm":
        run_vllm_evaluation(cfg, datasets)
    elif cfg.backend == "hf":
        run_hf_evaluation(cfg, datasets)
    else:
        raise ValueError(f"Invalid backend: {cfg.backend}. Must be 'vllm' or 'hf'.")
        
    logger.info(f"Evaluation complete! Results saved to {cfg.output_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )
    main()
