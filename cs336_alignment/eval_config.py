from dataclasses import dataclass, field
from typing import List, Optional

# --- Hydra Dataclasses for Config Type-Hinting ---
@dataclass
class ModelConfig:
    model_name_or_path: str
    dtype: str = "bfloat16"

@dataclass
class DatasetConfig:
    type: str
    data_path: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_split: Optional[str] = None
    num_samples: Optional[int] = None

@dataclass
class GenerationConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 32768
    stop: List[str] = field(default_factory=lambda: ["</answer>"])
    include_stop_str_in_output: bool = True

@dataclass
class VLLMConfig:
    num_gpus: int = 1

@dataclass
class ScriptArguments:
    backend: str
    model: ModelConfig
    dataset: DatasetConfig
    prompt_name: str
    generation: GenerationConfig
    vllm: VLLMConfig
    output_dir: str
    seed: int
