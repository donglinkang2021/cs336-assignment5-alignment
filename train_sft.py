import debugpy; debugpy.connect(('0.0.0.0', 5678))
"""
Train Qwen 2.5 Math 1.5B using Supervised Fine-Tuning (SFT) on OMR12k dataset.

This script implements the full SFT procedure to finetune the model on mathematical reasoning tasks.
It includes:
1. Loading the model and tokenizer
2. Loading the SFT dataset
3. Training with periodic evaluation
4. Logging to wandb
5. Saving checkpoints

Usage:
    python scripts/train_sft.py \
        --model-name-or-path models/Qwen2.5-Math-1.5B \
        --train-data-path data/OMR12k/train.jsonl \
        --val-data-path data/OMR12k/validation.jsonl \
        --output-dir outputs/sft_omr12k \
        --num-train-examples 12000 \
        --batch-size 4 \
        --gradient-accumulation-steps 4 \
        --learning-rate 5e-6 \
        --num-epochs 3 \
        --eval-steps 500 \
        --save-steps 1000 \
        --max-eval-examples 100 \
        --seed 42 \
        --use-wandb
"""

import os
import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from vllm import SamplingParams
from vllm_sync import VLLMClient

from cs336_alignment.sft_utils import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
    log_generations,
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

logger = logging.getLogger(__name__)


def load_prompt_template(prompt_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = Path("cs336_alignment") / "prompts" / f"{prompt_name}.prompt"
    with open(prompt_path) as f:
        return f.read()


class SFTDataset(Dataset):
    """Dataset for SFT training."""
    
    def __init__(self, data_path: str, prompt_template: str, num_examples: int = None):
        """
        Args:
            data_path: Path to the jsonl file containing the data
            prompt_template: Template for formatting prompts
            num_examples: Maximum number of examples to use (None = use all)
        """
        logger.info(f"Loading SFT data from {data_path}...")
        dataset = load_dataset('json', data_files=data_path, split='train')
        
        if num_examples is not None and num_examples < len(dataset):
            # Sample num_examples randomly
            indices = random.sample(range(len(dataset)), num_examples)
            dataset = dataset.select(indices)
        
        self.dataset = dataset
        self.prompt_template = prompt_template
        logger.info(f"Loaded {len(self.dataset)} examples for SFT")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return {
            "prompt": self.prompt_template.format(question=self.dataset[idx].get("problem", "")),
            "response": self.dataset[idx].get("generated_solution", ""),
            "ground_truth": self.dataset[idx].get("expected_answer", ""),
        }


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

def evaluate_model(
    vllm_client: VLLMClient,
    tokenizer: AutoTokenizer,
    val_dataset: SFTDataset,
    generation_kwargs: dict,
    max_examples: int = 100,
) -> Dict:
    """
    Evaluate the model on validation set using VLLMClient.
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Sample a subset for evaluation
    num_eval = min(max_examples, len(val_dataset))
    eval_indices = random.sample(range(len(val_dataset)), num_eval)
    
    prompts = [val_dataset[i]['prompt'] for i in eval_indices]
    ground_truths = [val_dataset[i]['ground_truth'] for i in eval_indices]
    
    logger.info(f"Generating responses for {len(prompts)} validation examples...")
    
    # Generate responses using VLLMClient
    output = vllm_client.generate(
        prompts, n=1, generation_kwargs=generation_kwargs
    )
    
    # Extract generated texts
    responses = []
    for completion_ids in output["completion_ids"]:
        # Decode each completion
        text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        responses.append(text)
    
    # Evaluate responses
    all_metrics = []
    for response, ground_truth in zip(responses, ground_truths):
        metrics = r1_zero_reward_fn(response, ground_truth)
        all_metrics.append(metrics)
    
    # Calculate aggregate metrics
    aggregate_metrics = {}
    for key in sorted(all_metrics[0].keys()):
        aggregate_metrics[key] = sum([m[key] for m in all_metrics]) / len(all_metrics)
    
    # Calculate accuracy (correct answers)
    accuracy = sum([m["answer_reward"] for m in all_metrics]) / len(all_metrics)
    aggregate_metrics["accuracy"] = accuracy
    
    return aggregate_metrics


def train(args):
    """Main training function."""
    
    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup wandb if enabled
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
        
        # Setup wandb metrics
        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model on GPU 0
    logger.info(f"Loading model from {args.model_name_or_path} on cuda:0...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=args.dtype,
        use_cache = not args.gradient_checkpointing,
    ).to("cuda:0")
    model.train()

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing to save memory...")
        model.gradient_checkpointing_enable()
    
    # Initialize VLLMClient for evaluation (if we have 2 GPUs)
    logger.info("Initializing VLLMClient for evaluation...")
    vllm_client = VLLMClient()
    vllm_client.init_communicator()
    
    # Load prompt template
    prompt_template = load_prompt_template("r1_zero")
    
    # Load datasets
    train_dataset = SFTDataset(
        args.train_data_path,
        prompt_template,
        num_examples=args.num_train_examples,
    )
    val_dataset = SFTDataset(
        args.val_data_path,
        prompt_template,
        num_examples=None,  # Use all validation examples
    )
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )
    
    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Calculate total training steps
    num_training_steps = len(train_loader) * args.num_epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    
    # Setup learning rate scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # Setup sampling parameters for evaluation
    generation_kwargs = dict(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        # stop=["</answer>"],
        # include_stop_str_in_output=True,
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    global_step = 0
    eval_step = 0
    best_accuracy = 0.0
    
    logger.info("Starting training...")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_epochs}")
    logger.info(f"  Batch size = {args.batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {num_training_steps // args.gradient_accumulation_steps}")
    
    for epoch in range(args.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
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
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                normalize_constant=response_mask.sum(dim=1).float().mean().item(),
            )
            
            epoch_loss += loss.item()
            
            # Gradient accumulation
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # Update weights
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Log training metrics
                avg_entropy = (token_entropy * response_mask).sum() / response_mask.sum()
                
                metrics = {
                    "train/loss": loss.item() * args.gradient_accumulation_steps,
                    "train/learning_rate": lr_scheduler.get_last_lr()[0],
                    "train/avg_token_entropy": avg_entropy.item(),
                    "train_step": global_step,
                }
                
                if args.use_wandb:
                    wandb.log(metrics)
                
                progress_bar.set_postfix({
                    "loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                })
                
                # Evaluation
                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    logger.info(f"Running evaluation at step {global_step}...")
                    
                    # Load current policy weights into VLLMClient
                    vllm_client.update_model_params(model)
                    
                    # Evaluate using VLLMClient
                    eval_metrics = evaluate_model(
                        vllm_client, tokenizer, val_dataset, generation_kwargs, 
                        args.max_eval_examples,
                    )
                    
                    eval_step += 1
                    
                    # Log evaluation metrics
                    eval_log = {
                        "eval/accuracy": eval_metrics["accuracy"],
                        "eval/reward": eval_metrics["reward"],
                        "eval/format_reward": eval_metrics["format_reward"],
                        "eval/answer_reward": eval_metrics["answer_reward"],
                        "eval_step": eval_step,
                    }
                    
                    if args.use_wandb:
                        wandb.log(eval_log)
                    
                    logger.info(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
                    logger.info(f"  Reward: {eval_metrics['reward']:.4f}")
                    
                    # Save best model
                    if eval_metrics["accuracy"] > best_accuracy:
                        best_accuracy = eval_metrics["accuracy"]
                        best_model_dir = output_dir / "best_model"
                        logger.info(f"New best accuracy: {best_accuracy:.4f}. Saving to {best_model_dir}")
                        model.save_pretrained(best_model_dir)
                        tokenizer.save_pretrained(best_model_dir)

                # Save checkpoint
                if args.save_steps > 0 and global_step % args.save_steps == 0:
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
    
    if args.use_wandb:
        wandb.finish()
    
    logger.info(f"Training completed! Best accuracy: {best_accuracy:.4f}")


def main():
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    
    parser = argparse.ArgumentParser(description="SFT training script")
    
    # Model arguments
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="models/Qwen2.5-Math-1.5B",
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Data type for model weights (e.g., float16, bfloat16)",
    )
    
    # Data arguments
    parser.add_argument(
        "--train-data-path",
        type=str,
        default="data/OMR12k/train.jsonl",
        help="Path to training data (jsonl format)",
    )
    parser.add_argument(
        "--val-data-path",
        type=str,
        default="data/OMR12k/validation.jsonl",
        help="Path to validation data (jsonl format)",
    )
    parser.add_argument(
        "--num-train-examples",
        type=int,
        default=None,
        help="Number of training examples to use (None = use all)",
    )
    
    # Training arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/sft_omr12k",
        help="Directory to save model and checkpoints",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="cosine",
        help="Learning rate scheduler type",
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Run evaluation every N steps (0 = no evaluation)",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=1000,
        help="Save checkpoint every N steps (0 = no checkpoints)",
    )
    parser.add_argument(
        "--max-eval-examples",
        type=int,
        default=100,
        help="Maximum number of examples to use for evaluation",
    )
    parser.add_argument(
        "--use-vllm-eval",
        action="store_true",
        help="Use vLLM for evaluation (requires 2 GPUs)",
    )
    
    # Logging arguments
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Use wandb for logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="cs336-sft",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Wandb run name",
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory (trades compute for memory)",
    )
    
    args = parser.parse_args()
    
    # Set default wandb run name if not provided
    if args.wandb_run_name is None:
        args.wandb_run_name = f"sft_{args.num_train_examples if args.num_train_examples else 'full'}"
    
    logger.info("Running: %s", " ".join(sys.argv))
    train(args)
    logger.info("Finished running %s", sys.argv[0])


if __name__ == "__main__":
    main()
