"""
Configuration for Task 2: LLM Alignment
Optimized for RTX 5090 on Windows 11
"""

from dataclasses import dataclass, field
from typing import Optional

# ============================================================================
# HARDWARE CONFIGURATION (RTX 5090)
# ============================================================================
DEVICE = "cuda"
DEVICE_MAP = "auto"
MAX_MEMORY = {0: "22GB"}  # RTX 5090 has 32GB, reserve some for OS
TORCH_DTYPE = "float16"
LOAD_IN_8BIT = True  # Use 8-bit quantization to reduce memory
LOAD_IN_4BIT = False  # Not needed with 8-bit

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-SFT-only"
REFERENCE_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-SFT-only"

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
DATASET_NAME = "Intel/orca_dpo_pairs"
DATASET_SPLIT = "train"
MAX_SAMPLES = 10000  # Limit for faster experimentation
EVAL_SIZE = 0.1
TEST_PROMPTS_NUM = 50

# ============================================================================
# LORA CONFIGURATION (PEFT)
# ============================================================================
USE_LORA = True
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_BIAS = "none"

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
@dataclass
class DPOConfig:
    """DPO Training Configuration"""
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-4
    beta: float = 0.1  # DPO temperature
    max_length: int = 512
    max_prompt_length: int = 256
    warmup_steps: int = 100
    weight_decay: float = 0.01
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50
    output_dir: str = "checkpoints/dpo"
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_32bit"


@dataclass
class PPORewardModelConfig:
    """Reward Model Configuration - Fine-tune on ORCA dataset"""
    model_id: str = "OpenAssistant/reward-model-deberta-v3-large"
    use_pretrained: bool = False  # Train on ORCA preference data by default
    num_train_epochs: int = 2  # 2 epochs for fine-tuning large model
    per_device_train_batch_size: int = 8  # Smaller batch for large model
    per_device_eval_batch_size: int = 16
    learning_rate: float = 2e-5  # Lower LR for fine-tuning
    max_length: int = 512
    warmup_steps: int = 50
    weight_decay: float = 0.01
    output_dir: str = "checkpoints/reward_model"
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_32bit"
    load_in_8bit: bool = False  # Disabled - RTX 5090 has enough VRAM


@dataclass
class PPOConfig:
    """PPO Training Configuration"""
    num_ppo_epochs: int = 4
    per_device_train_batch_size: int = 1  # Reduced from 4 for debugging
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    value_learning_rate: float = 1e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 1.0
    max_length: int = 512
    max_prompt_length: int = 256
    kl_coeff: float = 0.05
    warmup_steps: int = 100
    output_dir: str = "checkpoints/ppo"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_32bit"
    # Sparse vs Dense rewards
    reward_type: str = "sparse"  # "sparse" or "dense"


@dataclass
class GRPOConfig:
    """GRPO Training Configuration"""
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1  # Reduced from 4 for debugging
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    group_size: int = 2  # Reduced from 4 for debugging - number of responses per prompt
    max_length: int = 512
    max_prompt_length: int = 256
    kl_coeff: float = 0.05
    warmup_steps: int = 100
    output_dir: str = "checkpoints/grpo"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_32bit"


# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
EVAL_REWARD_MODEL_ID = "OpenAssistant/reward-model-deberta-v3-large"
TEST_PROMPTS_FILE = "data/test_prompts.json"
HACK_PROMPTS_FILE = "data/hack_prompts.json"

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================
RESULTS_DIR = "results"
CHECKPOINTS_DIR = "checkpoints"
VISUALIZATIONS_DIR = "visualizations"
SAMPLES_DIR = "samples"

# Logging
SAVE_JSON_RESULTS = True
SAVE_PLOTS = True
SAVE_CHECKPOINTS = True
SAVE_SAMPLES = True
SAVE_GENERATIONS = True
