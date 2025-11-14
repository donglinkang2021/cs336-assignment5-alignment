"""
Configuration dataclasses for SFT training script using Hydra.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name_or_path: str = "models/Qwen2.5-Math-1.5B"
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = False
    attn_implementation: str = "flash_attention_2"


@dataclass
class DataConfig:
    """Data configuration."""
    train_data_path: str = "data/OMR12k/train.jsonl"
    val_data_path: str = "data/OMR12k/validation.jsonl"
    num_train_examples: Optional[int] = None
    prompt_name: str = "r1_zero"


@dataclass
class TrainingConfig:
    """Training configuration."""
    output_dir: str = "outputs/sft_omr12k"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    seed: int = 42


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    eval_steps: int = 500
    save_steps: int = 1000
    max_eval_examples: int = 100
    use_vllm_eval: bool = True
    generation_temperature: float = 1.0
    generation_top_p: float = 1.0
    generation_max_tokens: int = 32768


@dataclass
class LoggingConfig:
    """Logging configuration."""
    use_wandb: bool = False
    wandb_project: str = "cs336-align"
    wandb_run_name: Optional[str] = None


@dataclass
class ScriptArguments:
    """Main configuration container for SFT training."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
