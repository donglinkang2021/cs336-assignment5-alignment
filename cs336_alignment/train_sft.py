# import debugpy; debugpy.connect(('0.0.0.0', 5678))
"""
Train Qwen 2.5 Math 1.5B using Supervised Fine-Tuning (SFT) on OMR12k dataset.

This script implements the full SFT procedure to finetune the model on mathematical reasoning tasks.
It includes:
1. Loading the model and tokenizer
2. Loading the SFT dataset
3. Training with periodic evaluation
4. Logging to wandb
5. Saving checkpoints
"""

import time
import logging
import random
from pathlib import Path
from typing import Dict, List

import wandb
import hydra
import torch
from datasets import load_dataset, Dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from cs336_alignment.vllm_sync import VLLMClient

from cs336_alignment.sft_utils import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.optimizer import MemoryEfficientAdamW
from cs336_alignment.config.sft_config import ScriptArguments
from cs336_alignment.utils import load_prompt_template

logger = logging.getLogger(__name__)

def process_math_example(sample: Dict, prompt_template: str) -> Dict:
    """
    Process a single MATH example by formatting the prompt.
    """
    question = sample.get("problem", sample.get("question", ""))
    response = sample.get("generated_solution", "").lstrip("<think>\n")
    answer = sample.get("solution", sample.get("expected_answer", sample.get("answer", "")))
    sample["prompt"] = prompt_template.format(question=question)
    sample["response"]=response
    sample["ground_truth"] = answer
    return sample

def sample_dataset(dataset: Dataset, num_samples:int=None) -> Dataset:
    if num_samples is not None and num_samples < len(dataset):
        num_samples = min(num_samples, len(dataset))
        logger.info(f"Sampling {num_samples} examples from the dataset.")
        indices = random.sample(range(len(dataset)), num_samples)
        dataset = dataset.select(indices)
    return dataset

def get_dataset(dataset_cfg: dict, prompt_template: str, num_samples: int = None) -> Dataset:
    logger.info(f"Loading data from {dataset_cfg.get("data_files", "")}...")
    dataset = load_dataset(**dataset_cfg)
    dataset = sample_dataset(dataset, num_samples)
    dataset = dataset.map(
        lambda sample: process_math_example(sample, prompt_template),
        desc="Formatting prompts"
    )
    logger.info(f"Loaded {len(dataset)} examples.")
    return dataset

def collate_fn(batch: List[Dict], tokenizer):
    """Collate function for DataLoader."""
    prompts = [item["prompt"] for item in batch]
    responses = [item["response"] for item in batch]
    ground_truths = [item["ground_truth"] for item in batch]
    
    # Tokenize prompts and responses
    tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
    
    return {
        "input_ids": tokenized["input_ids"],
        "labels": tokenized["labels"],
        "response_mask": tokenized["response_mask"],
        "prompts": prompts,
        "responses": responses,
        "ground_truths": ground_truths,
    }

def mean(values: List[float]) -> float:
    """Calculate mean of a list of values."""
    return sum(values) / len(values) if values else 0.0

def evaluate_model(
    vllm_client: VLLMClient,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    generation_kwargs: dict,
    eval_n_times: int
) -> Dict:    
    logger.info(f"Generating responses for {len(dataset)} validation examples...")
    output = vllm_client.generate(dataset["prompt"], n=eval_n_times, generation_kwargs=generation_kwargs)
    responses = tokenizer.batch_decode(output["completion_ids"], skip_special_tokens=True)
    
    # Evaluate responses and collect detailed metrics
    all_metrics = []
    
    for response, ground_truth in zip(responses, dataset["ground_truth"]):
        metrics = r1_zero_reward_fn(response, ground_truth)
        all_metrics.append(metrics)
    
    # Calculate aggregate metrics
    aggregate_metrics = {}
    for key in sorted(all_metrics[0].keys()):
        aggregate_metrics[key] = mean([m[key] for m in all_metrics])
    
    # Calculate accuracy (correct answers)
    aggregate_metrics["accuracy"] = aggregate_metrics["answer_reward"]
    
    # Calculate response length statistics
    response_lengths = [len(r) for r in responses]
    correct_lengths = [len(responses[i]) for i, m in enumerate(all_metrics) if m["answer_reward"] > 0]
    incorrect_lengths = [len(responses[i]) for i, m in enumerate(all_metrics) if m["answer_reward"] <= 0]
    
    aggregate_metrics["avg_response_length/total"] = mean(response_lengths)
    aggregate_metrics["avg_response_length/correct"] = mean(correct_lengths)
    aggregate_metrics["avg_response_length/incorrect"] = mean(incorrect_lengths)
    
    return aggregate_metrics

def log_eval_metrics(eval_metrics:Dict, use_wandb:bool, eval_step:int, dataset_name:str):
    if use_wandb:
        # Log evaluation metrics with dataset name prefix
        wandb.log({
            f"eval/{dataset_name}/accuracy": eval_metrics["accuracy"],
            f"eval/{dataset_name}/reward": eval_metrics["reward"],
            f"eval/{dataset_name}/format_reward": eval_metrics["format_reward"],
            f"eval/{dataset_name}/answer_reward": eval_metrics["answer_reward"],
            f"eval/{dataset_name}/avg_response_length/total": eval_metrics["avg_response_length/total"],
            f"eval/{dataset_name}/avg_response_length/correct": eval_metrics["avg_response_length/correct"],
            f"eval/{dataset_name}/avg_response_length/incorrect": eval_metrics["avg_response_length/incorrect"],
            "eval_step": eval_step,
        })
    
    logger.info(f"  [{dataset_name}] Accuracy: {eval_metrics['accuracy']:.4f}")
    logger.info(f"  [{dataset_name}] Reward: {eval_metrics['reward']:.4f}")
    logger.info(f"  [{dataset_name}] Avg Response Length: {eval_metrics['avg_response_length/total']:.1f}")
    logger.info(f"  [{dataset_name}] Avg Length (Correct): {eval_metrics['avg_response_length/correct']:.1f}")
    logger.info(f"  [{dataset_name}] Avg Length (Incorrect): {eval_metrics['avg_response_length/incorrect']:.1f}")


@hydra.main(version_base=None, config_path="../conf", config_name="sft_config")
def train(cfg: ScriptArguments):
    """Main training function."""
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    # Log configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Set random seeds
    random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training.seed)
    
    # Setup wandb if enabled
    if cfg.logging.use_wandb:
        
        # Set default wandb run name if not provided
        wandb_run_name = cfg.logging.wandb_run_name
        if wandb_run_name is None:
            wandb_run_name = f"sft_{cfg.data.num_train_examples if cfg.data.num_train_examples else 'full'}"
        
        wandb.init(
            project=cfg.logging.wandb_project,
            name=wandb_run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        
        # Setup wandb metrics
        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {cfg.model.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model on GPU 0
    logger.info(f"Loading model from {cfg.model.model_name_or_path} on cuda:0...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name_or_path,
        dtype=cfg.model.dtype,
        attn_implementation=cfg.model.attn_implementation,
        use_cache = not cfg.model.gradient_checkpointing,
    ).to("cuda:0")
    model.train()

    # Enable gradient checkpointing if requested
    if cfg.model.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing to save memory...")
        model.gradient_checkpointing_enable()
    
    # Initialize VLLMClient for evaluation (if we have 2 GPUs)
    logger.info("Initializing VLLMClient for evaluation...")
    vllm_client = VLLMClient()
    vllm_client.init_communicator()
    
    # Load prompt template
    prompt_template = load_prompt_template(cfg.data.prompt_name)
    
    # Load train dataset
    train_dataset = get_dataset({
        'path': "json", 'data_files': cfg.data.train_data_path, 'split': "train"
    }, prompt_template, cfg.data.num_train_examples)
    
    # Load multiple validation datasets
    logger.info("Loading validation datasets...")
    val_datasets = {}
    for dataset_name, dataset_cfg in cfg.val_datasets.items():
        logger.info(f"Loading validation dataset: {dataset_name}")
        val_dataset = get_dataset(dataset_cfg, prompt_template, cfg.evaluation.max_eval_examples)
        val_datasets[dataset_name] = val_dataset
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )
    
    # Setup optimizer
    optimizer = MemoryEfficientAdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    
    # Calculate total training steps
    num_training_steps = len(train_loader) * cfg.training.num_epochs // cfg.training.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * cfg.training.warmup_ratio)
    
    # Setup learning rate scheduler
    lr_scheduler = get_scheduler(
        name=cfg.training.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # Setup sampling parameters for evaluation
    generation_kwargs = dict(
        temperature=cfg.evaluation.generation_temperature,
        top_p=cfg.evaluation.generation_top_p,
        max_tokens=cfg.evaluation.generation_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    
    # Create output directory
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    global_step = 0
    eval_step = 0
    best_accuracy = 0.0
    consumed_tokens = 0  # Track total tokens consumed during training
    start_time = None  # Track start time for tokens/sec calculation
    
    logger.info("Starting training...")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {cfg.training.num_epochs}")
    logger.info(f"  Batch size = {cfg.training.batch_size}")
    logger.info(f"  Gradient accumulation steps = {cfg.training.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {num_training_steps}")
    
    for epoch in range(cfg.training.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{cfg.training.num_epochs}")
        
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            # Start timing on first step
            if start_time is None:
                start_time = time.time()
            
            # Move batch to device
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)
            response_mask = batch["response_mask"].to(model.device)
            
            # Forward pass
            output = get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=True,
            )
            
            policy_log_probs = output["log_probs"]
            token_entropy = output["token_entropy"]
            
            # Calculate loss
            loss, _ = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
                normalize_constant=response_mask.sum(dim=1).float().mean().item(),
            )
            
            epoch_loss += loss.item()
            
            # Count tokens in this microbatch (only count response tokens where we compute loss)
            batch_tokens = response_mask.sum().item()
            consumed_tokens += batch_tokens
            
            # Gradient accumulation
            if (step + 1) % cfg.training.gradient_accumulation_steps == 0:
                # Clip gradients and get gradient norm
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                
                # Update weights
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Calculate tokens per second
                elapsed_time = time.time() - start_time
                tokens_per_sec = consumed_tokens / elapsed_time if elapsed_time > 0 else 0
                
                # Log training metrics
                avg_entropy = (token_entropy * response_mask).sum() / response_mask.sum()
                
                metrics = {
                    "train/loss": loss.item() * cfg.training.gradient_accumulation_steps,
                    "train/learning_rate": lr_scheduler.get_last_lr()[0],
                    "train/avg_token_entropy": avg_entropy.item(),
                    "train/consumed_tokens": consumed_tokens,
                    "train/grad_norm": grad_norm.item(),
                    "train/tokens_per_sec": tokens_per_sec,
                    "train_step": global_step,
                }
                
                if cfg.logging.use_wandb:
                    wandb.log(metrics)
                
                progress_bar.set_postfix({
                    "loss": f"{loss.item() * cfg.training.gradient_accumulation_steps:.4f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    "tokens": f"{consumed_tokens/1e6:.2f}M",
                    "tok/s": f"{tokens_per_sec:.0f}",
                    "grad": f"{grad_norm.item():.2f}",
                })
                
                # Evaluation
                if cfg.evaluation.eval_steps > 0 and global_step % cfg.evaluation.eval_steps == 0:
                    logger.info(f"Running evaluation at step {global_step}...")
                    
                    # Load current policy weights into VLLMClient
                    vllm_client.update_model_params(model)
                    
                    eval_step += 1
                    
                    # Evaluate on all validation datasets
                    all_eval_metrics = {}
                    for dataset_name, val_dataset in val_datasets.items():
                        logger.info(f"Evaluating on {dataset_name}...")
                        eval_metrics = evaluate_model(
                            vllm_client, tokenizer, val_dataset,
                            generation_kwargs, cfg.evaluation.eval_n_times,
                        )
                        all_eval_metrics[dataset_name] = eval_metrics
                        log_eval_metrics(eval_metrics, cfg.logging.use_wandb, eval_step, dataset_name)
                    
                    # Save best model based on first dataset's accuracy
                    first_dataset_name = list(val_datasets.keys())[0]
                    first_dataset_accuracy = all_eval_metrics[first_dataset_name]["accuracy"]
                    
                    if first_dataset_accuracy > best_accuracy:
                        best_accuracy = first_dataset_accuracy
                        best_model_dir = output_dir / "best_model"
                        logger.info(f"New best accuracy on {first_dataset_name}: {best_accuracy:.4f}. Saving to {best_model_dir}")
                        model.save_pretrained(best_model_dir)
                        tokenizer.save_pretrained(best_model_dir)

                # Save checkpoint
                if cfg.evaluation.save_steps > 0 and global_step % cfg.evaluation.save_steps == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    logger.info(f"Saving checkpoint to {checkpoint_dir}")
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
        
        # Log epoch metrics
        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
    
    # Save final model
    final_model_dir = output_dir / "final_model"
    logger.info(f"Saving final model to {final_model_dir}")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    if cfg.logging.use_wandb:
        wandb.finish()
    
    logger.info(f"Training completed! Best accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    train()
