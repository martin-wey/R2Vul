from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ModelArguments:
    model_name_or_path: str = "Qwen/Qwen2.5-Coder-32B-Instruct"
    checkpoint_path: Optional[str] = None
    completion_only: Optional[bool] = False
    attn_implementation: Optional[str] = "flash_attention_2"

    use_qlora: Optional[bool] = False
    lora_r: Optional[int] = 8
    lora_alpha: Optional[int] = 16
    lora_dropout: Optional[float] = 0.05


@dataclass
class DataArguments:
    data_dir: Optional[str] = "data"
    dataset_name: Optional[str] = "r2vul_dataset"
    languages: Optional[List[str]] = None

    # reverse label for negative sample reasoning generation
    reverse_labels: Optional[bool] = False
    preprocessing_num_workers: Optional[int] = 32


@dataclass
class InferenceArguments:
    model_name_or_path: str = "Qwen/Qwen2.5-Coder-32B-Instruct"
    output_dir: Optional[str] = None

    max_new_tokens: Optional[int] = 2048
    strategy: Optional[str] = "zero-shot"
    batch_size: int = 32
    test_ratio: Optional[str] = "1:1"

    do_sample: Optional[bool] = False
    temperature: Optional[float] = 0.2
    top_k: Optional[float] = 40
    top_p: Optional[float] = 0.95
    num_return_sequences: Optional[int] = 10
