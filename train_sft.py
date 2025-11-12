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

Usage:
    python train_sft.py
    
    # Override config values from command line
    python train_sft.py training.num_epochs=5 training.batch_size=8
    
    # Use wandb
    python train_sft.py logging.use_wandb=true logging.wandb_run_name=my_experiment
"""

import logging
import random
from pathlib import Path
from typing import Dict, List

import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from vllm_sync import VLLMClient

from cs336_alignment.sft_utils import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.optimizer import MemoryEfficientAdamW
from cs336_alignment.sft_config import ScriptArguments

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
        prompt = self.prompt_template.format(question=self.dataset[idx].get("problem", ""))
        response = self.dataset[idx].get("generated_solution", "")
        ground_truth = self.dataset[idx].get("expected_answer", "")
        return { "prompt": prompt, "response": response, "ground_truth": ground_truth}


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


def train(cfg: ScriptArguments):
    """Main training function."""
    
    # Set random seeds
    random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training.seed)
    
    # Setup wandb if enabled
    if cfg.logging.use_wandb:
        import wandb
        
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
    
    # Load datasets
    train_dataset = SFTDataset(
        cfg.data.train_data_path,
        prompt_template,
        num_examples=cfg.data.num_train_examples,
    )
    val_dataset = SFTDataset(
        cfg.data.val_data_path,
        prompt_template,
        num_examples=None,  # Use all validation examples
    )
    
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
                gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
                normalize_constant=response_mask.sum(dim=1).float().mean().item(),
            )
            
            epoch_loss += loss.item()
            
            # Gradient accumulation
            if (step + 1) % cfg.training.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                
                # Update weights
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Log training metrics
                avg_entropy = (token_entropy * response_mask).sum() / response_mask.sum()
                
                metrics = {
                    "train/loss": loss.item() * cfg.training.gradient_accumulation_steps,
                    "train/learning_rate": lr_scheduler.get_last_lr()[0],
                    "train/avg_token_entropy": avg_entropy.item(),
                    "train_step": global_step,
                }
                
                if cfg.logging.use_wandb:
                    wandb.log(metrics)
                
                progress_bar.set_postfix({
                    "loss": f"{loss.item() * cfg.training.gradient_accumulation_steps:.4f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                })
                
                # Evaluation
                if cfg.evaluation.eval_steps > 0 and global_step % cfg.evaluation.eval_steps == 0:
                    logger.info(f"Running evaluation at step {global_step}...")
                    
                    # Load current policy weights into VLLMClient
                    vllm_client.update_model_params(model)
                    
                    # Evaluate using VLLMClient
                    eval_metrics = evaluate_model(
                        vllm_client, tokenizer, val_dataset, generation_kwargs, 
                        cfg.evaluation.max_eval_examples,
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
                    
                    if cfg.logging.use_wandb:
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


@hydra.main(version_base=None, config_path="conf", config_name="sft_config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    
    # Convert DictConfig to ScriptArguments dataclass
    script_args:ScriptArguments = hydra.utils.instantiate(cfg, _convert_="object")
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Run training
    train(script_args)


if __name__ == "__main__":
    main()
