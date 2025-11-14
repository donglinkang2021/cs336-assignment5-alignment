"""
Test script to find maximum batch size for SFT training.
This script tests different batch sizes and measures peak GPU memory usage.
"""

import torch
import logging
import json
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from tabulate import tabulate
from cs336_alignment.optimizer import MemoryEfficientAdamW
from cs336_alignment.sft_utils import get_response_log_probs, sft_microbatch_train_step

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_batch_size(
    model,
    tokenizer,
    batch_size: int,
    seq_length: int = 2048,
    use_memory_efficient_optimizer: bool = True,
    gradient_checkpointing: bool = True,
    use_flash_attention: bool = True,
):
    """
    Test if a given batch size fits in GPU memory.
    
    Args:
        model: The model to test
        tokenizer: Tokenizer for the model
        batch_size: Batch size to test
        seq_length: Sequence length for generated data
        use_memory_efficient_optimizer: Whether to use MemoryEfficientAdamW
        gradient_checkpointing: Whether to use gradient checkpointing
        use_flash_attention: Whether to use Flash Attention
    
    Returns:
        tuple: (success: bool, peak_memory_mb: float)
    """
    try:
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Set gradient checkpointing
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()
        
        # Setup optimizer
        if use_memory_efficient_optimizer:
            optimizer = MemoryEfficientAdamW(
                model.parameters(),
                lr=5e-6,
                weight_decay=0.01,
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=5e-6,
                weight_decay=0.01,
            )
        
        # Generate random data
        input_ids = torch.randint(
            0, tokenizer.vocab_size, 
            (batch_size, seq_length+1),
            device=model.device
        )
        
        # Create labels (same as input_ids for simplicity)
        labels = input_ids[:,:-1]
        input_ids = input_ids[:,1:]        
        
        # Create response mask (assume last 1024 tokens are response)
        response_mask = torch.ones_like(input_ids, device=model.device)
        
        # Perform multiple iterations to see the memory difference
        # After first iteration, optimizer states are initialized
        # MemoryEfficientAdamW keeps them on CPU, AdamW keeps them on GPU
        num_iterations = 3
        
        for iteration in range(num_iterations):
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=True,
            )
            
            policy_log_probs = output["log_probs"]
            
            # Calculate loss (backward is called internally)
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=1,
                normalize_constant=response_mask.sum(dim=1).float().mean().item(),
            )
            
            # Optimizer step (backward was already called in sft_microbatch_train_step)
            optimizer.step()
            
            # Force CUDA synchronization to ensure memory is allocated
            torch.cuda.synchronize()
        
        # Get peak memory
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
        
        # Calculate optimizer state memory (for logging)
        optimizer_state_memory = 0
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    # Check if tensor is on GPU
                    if value.device.type == 'cuda':
                        optimizer_state_memory += value.numel() * value.element_size()
        optimizer_state_memory_mb = optimizer_state_memory / 1024**2
        
        logger.info(f"✓ Batch size {batch_size} succeeded. Peak memory: {peak_memory:.2f} MB, "
                   f"Optimizer state on GPU: {optimizer_state_memory_mb:.2f} MB")
        
        return True, peak_memory
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.info(f"✗ Batch size {batch_size} failed: OOM")
            torch.cuda.empty_cache()
            return False, 0.0
        else:
            raise e
    finally:
        # Cleanup
        del optimizer
        torch.cuda.empty_cache()


def find_max_batch_size(
    model_name_or_path: str,
    dtype: str = "bfloat16",
    use_memory_efficient_optimizer: bool = True,
    gradient_checkpointing: bool = True,
    use_flash_attention: bool = True,
    max_batch_size: int = 64,
    seq_length: int = 2048,
):
    """
    Binary search to find maximum batch size.
    
    Args:
        model_name_or_path: Path to model
        dtype: Data type for model
        use_memory_efficient_optimizer: Whether to use MemoryEfficientAdamW
        gradient_checkpointing: Whether to use gradient checkpointing
        use_flash_attention: Whether to use Flash Attention
        max_batch_size: Maximum batch size to test
        seq_length: Sequence length for testing
    """
    logger.info(f"Loading model from {model_name_or_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with Flash Attention configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=getattr(torch, dtype),
        use_cache=not gradient_checkpointing,
        attn_implementation="flash_attention_2" if use_flash_attention else "eager",
    ).to("cuda")
    
    model.train()
    
    optimizer_type = "MemoryEfficientAdamW" if use_memory_efficient_optimizer else "AdamW"
    gc_status = "enabled" if gradient_checkpointing else "disabled"
    fa_status = "enabled" if use_flash_attention else "disabled"
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Configuration:")
    logger.info(f"  Model: {model_name_or_path}")
    logger.info(f"  Dtype: {dtype}")
    logger.info(f"  Optimizer: {optimizer_type}")
    logger.info(f"  Gradient Checkpointing: {gc_status}")
    logger.info(f"  Flash Attention: {fa_status}")
    logger.info(f"  Sequence Length: {seq_length}")
    logger.info(f"{'='*80}\n")
    
    # Binary search for max batch size
    left, right = 1, max_batch_size
    max_successful_bs = 0
    max_successful_memory = 0.0
    
    logger.info("Starting binary search for maximum batch size...\n")
    
    while left <= right:
        mid = (left + right) // 2
        logger.info(f"Testing batch size: {mid}")
        
        success, peak_memory = test_batch_size(
            model, tokenizer, mid, seq_length,
            use_memory_efficient_optimizer, gradient_checkpointing, use_flash_attention
        )
        
        if success:
            max_successful_bs = mid
            max_successful_memory = peak_memory
            left = mid + 1
        else:
            right = mid - 1
        
        logger.info("")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Results:")
    logger.info(f"  Maximum Batch Size: {max_successful_bs}")
    logger.info(f"  Peak Memory Usage: {max_successful_memory:.2f} MB ({max_successful_memory/1024:.2f} GB)")
    logger.info(f"  Configuration: {optimizer_type} + GradCheckpoint {gc_status} + FlashAttn {fa_status}")
    logger.info(f"{'='*80}\n")
    
    return max_successful_bs, max_successful_memory


def compare_configurations(model_name_or_path: str, seq_length = 4096, output_file: str = None):
    """
    Compare different optimizer and gradient checkpointing configurations.
    
    Args:
        model_name_or_path: Path to the model
        output_file: Path to save results as JSONL file. If None, uses default name.
    """
    results = []
    
    # Generate default output filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(model_name_or_path).name
        Path("batch_size_benchmark").mkdir(exist_ok=True)
        output_file = f"batch_size_benchmark/{model_name}-{seq_length}_{timestamp}.jsonl"
    
    # Get basic model info
    logger.info(f"Loading model to get basic information...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    configs = [
        ("MemoryEfficientAdamW", True, True, True),   # Memory efficient + GC + FA
        ("MemoryEfficientAdamW", True, True, False),  # Memory efficient + GC, no FA
        ("MemoryEfficientAdamW", True, False, True),  # Memory efficient + FA, no GC
        ("MemoryEfficientAdamW", True, False, False), # Memory efficient, no GC, no FA
        ("AdamW", False, True, True),                 # Standard AdamW + GC + FA
        ("AdamW", False, True, False),                # Standard AdamW + GC, no FA
        ("AdamW", False, False, True),                # Standard AdamW + FA, no GC
        ("AdamW", False, False, False),               # Standard AdamW, no GC, no FA
    ]
    
    for optimizer_name, use_mem_efficient, use_gc, use_fa in configs:
        logger.info(f"\n{'#'*80}")
        logger.info(f"Testing: {optimizer_name} + GradCheckpoint={'ON' if use_gc else 'OFF'} + FlashAttn={'ON' if use_fa else 'OFF'}")
        logger.info(f"{'#'*80}")
        
        max_bs, peak_mem = find_max_batch_size(
            model_name_or_path=model_name_or_path,
            dtype="bfloat16",
            use_memory_efficient_optimizer=use_mem_efficient,
            gradient_checkpointing=use_gc,
            use_flash_attention=use_fa,
            max_batch_size=64,
            seq_length=seq_length,
        )
        
        # Create detailed result entry
        result = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name_or_path,
            "model_short_name": Path(model_name_or_path).name,
            "sequence_length": seq_length,
            "dtype": "bfloat16",
            "vocab_size": tokenizer.vocab_size,
            "optimizer": optimizer_name,
            "gradient_checkpointing": use_gc,
            "flash_attention": use_fa,
            "max_batch_size": max_bs,
            "peak_memory_mb": peak_mem,
            "peak_memory_gb": peak_mem / 1024,
        }
        
        results.append(result)
        
        # Write to JSONL file incrementally
        with open(output_file, 'a') as f:
            f.write(json.dumps(result) + '\n')
        
        logger.info(f"Result saved to {output_file}")
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY OF ALL CONFIGURATIONS:")
    logger.info(f"{'='*80}\n")
    
    # Print metadata
    logger.info(f"Model: {model_name_or_path}")
    logger.info(f"Sequence Length: {seq_length}")
    logger.info(f"Vocab Size: {tokenizer.vocab_size}")
    logger.info(f"Results saved to: {output_file}\n")
    
    # Prepare table data
    table_data = []
    for result in results:
        gc_status = "ON" if result["gradient_checkpointing"] else "OFF"
        fa_status = "ON" if result["flash_attention"] else "OFF"
        table_data.append([
            result['optimizer'],
            gc_status,
            fa_status,
            result['max_batch_size'],
            f"{result['peak_memory_gb']:.2f}"
        ])
    
    # Print table
    headers = ["Optimizer", "Grad Checkpoint", "Flash Attention", "Max Batch Size", "Peak Memory (GB)"]
    table_str = tabulate(table_data, headers=headers, tablefmt="grid")
    logger.info(f"\n{table_str}\n")
    
    logger.info(f"{'='*80}\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test maximum batch size for SFT training")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/Qwen2.5-Math-1.5B",
        help="Path to model"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all optimizer/gradient checkpointing configurations"
    )
    parser.add_argument(
        "--memory_efficient",
        action="store_true",
        default=True,
        help="Use MemoryEfficientAdamW (default: True)"
    )
    parser.add_argument(
        "--no_memory_efficient",
        dest="memory_efficient",
        action="store_false",
        help="Use standard AdamW instead of MemoryEfficientAdamW"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing (default: True)"
    )
    parser.add_argument(
        "--no_gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable gradient checkpointing"
    )
    parser.add_argument(
        "--flash_attention",
        action="store_true",
        default=True,
        help="Enable Flash Attention 2 (default: True)"
    )
    parser.add_argument(
        "--no_flash_attention",
        dest="flash_attention",
        action="store_false",
        help="Disable Flash Attention 2"
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=4096,
        help="Sequence length for testing (default: 4096)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL file path (default: auto-generated with timestamp)"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare all configurations
        compare_configurations(args.model_path, args.seq_length, output_file=args.output)
    else:
        # Test single configuration
        result = find_max_batch_size(
            model_name_or_path=args.model_path,
            dtype="bfloat16",
            use_memory_efficient_optimizer=args.memory_efficient,
            gradient_checkpointing=args.gradient_checkpointing,
            use_flash_attention=args.flash_attention,
            max_batch_size=64,
            seq_length=args.seq_length,
        )
        
        # Save single result to file if output path is provided
        if args.output:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            max_bs, peak_mem = result
            single_result = {
                "timestamp": datetime.now().isoformat(),
                "model_name": args.model_path,
                "model_short_name": Path(args.model_path).name,
                "sequence_length": args.seq_length,
                "dtype": "bfloat16",
                "vocab_size": tokenizer.vocab_size,
                "optimizer": "MemoryEfficientAdamW" if args.memory_efficient else "AdamW",
                "gradient_checkpointing": args.gradient_checkpointing,
                "flash_attention": args.flash_attention,
                "max_batch_size": max_bs,
                "peak_memory_mb": peak_mem,
                "peak_memory_gb": peak_mem / 1024,
            }
            
            with open(args.output, 'w') as f:
                f.write(json.dumps(single_result) + '\n')
            
            logger.info(f"\nResult saved to {args.output}")