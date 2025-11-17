from dataclasses import dataclass, field
from typing import List, Optional

# --- Hydra Dataclasses for Config Type-Hinting ---
@dataclass
class ModelConfig:
    model_name_or_path: str
    dtype: str = "bfloat16"

@dataclass
class DatasetConfig:
    """Just the arguments of the datasets.load_dataset"""
    path: str
    name: Optional[str] = None
    data_files: Optional[str] = None
    split: Optional[str] = None

@dataclass
class GenerationConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 4096
    stop: List[str] = field(default_factory=lambda: ["</answer>"])
    include_stop_str_in_output: bool = True

@dataclass
class ScriptArguments:
    backend: str
    model: ModelConfig
    datasets: List[DatasetConfig]
    prompt_name: str
    generation: GenerationConfig
    output_dir: str
    seed: int
    num_gpus: int = 1
    num_samples: Optional[int] = None
