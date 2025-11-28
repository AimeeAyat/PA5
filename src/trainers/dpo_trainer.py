"""
DPO (Direct Preference Optimization) Trainer
Implements DPO as described in the PA5 motivation section
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import DPOTrainer

logger = logging.getLogger(__name__)


class DPOAlignmentTrainer:
    """DPO alignment trainer using TRL library"""

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolLM2-135M-SFT-only",
        ref_model_id: Optional[str] = None,
        output_dir: str = "checkpoints/dpo",
        load_in_8bit: bool = True,
        use_lora: bool = True,
    ):
        """
        Initialize DPO Trainer

        Args:
            model_id: Base model identifier
            ref_model_id: Reference model for KL penalty (if None, use model_id)
            output_dir: Directory to save checkpoints
            load_in_8bit: Use 8-bit quantization
            use_lora: Use LoRA adapters
        """
        self.model_id = model_id
        self.ref_model_id = ref_model_id or model_id
        self.output_dir = output_dir
        self.load_in_8bit = load_in_8bit
        self.use_lora = use_lora
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.trainer = None
        self.training_history = {}

    def load_models(self) -> None:
        """Load model and reference model"""
        logger.info(f"Loading base model: {self.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
        }

        if self.load_in_8bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
            )
            model_kwargs["quantization_config"] = bnb_config

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs
        )

        # Load reference model (typically same as base)
        logger.info(f"Loading reference model: {self.ref_model_id}")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.ref_model_id,
            **model_kwargs
        )

        # Apply LoRA if requested
        if self.use_lora:
            self._apply_lora()

        logger.info("Models loaded successfully")

    def _apply_lora(self) -> None:
        """Apply LoRA adapters to model"""
        logger.info("Applying LoRA adapters...")

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        logger.info("LoRA applied successfully")

    def preprocess_dpo_data(self, example):
        """Preprocess example for DPO training"""
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 16,
        learning_rate: float = 5e-4,
        beta: float = 0.1,
        max_length: int = 512,
        max_prompt_length: int = 256,
        warmup_steps: int = 100,
    
        weight_decay: float = 0.01,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 50,
    ) -> Dict:
        """
        Train model with DPO

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size
            per_device_eval_batch_size: Eval batch size
            learning_rate: Learning rate
            beta: DPO temperature parameter
            max_length: Maximum sequence length
            max_prompt_length: Maximum prompt length
            warmup_steps: Warmup steps
            weight_decay: Weight decay
            save_steps: Save checkpoint frequency
            eval_steps: Evaluation frequency
            logging_steps: Logging frequency

        Returns:
            Training metrics dictionary
        """
        if self.model is None:
            self.load_models()

        logger.info("Preprocessing datasets for DPO...")
        train_dataset = train_dataset.map(
            self.preprocess_dpo_data,
            remove_columns=[col for col in train_dataset.column_names if col not in ["prompt", "chosen", "rejected"]],
            desc="Processing train data"
        )
        eval_dataset = eval_dataset.map(
            self.preprocess_dpo_data,
            remove_columns=[col for col in eval_dataset.column_names if col not in ["prompt", "chosen", "rejected"]],
            desc="Processing eval data"
        )

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=2,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            save_steps=save_steps,
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            bf16=False,
            fp16=False,
            optim="paged_adamw_32bit",
            remove_unused_columns=False,
        )

        # Create DPOConfig for newer TRL versions
        from trl import DPOConfig
        
        dpo_config = DPOConfig(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=2,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            save_steps=save_steps,
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            bf16=False,
            fp16=False,
            optim="paged_adamw_32bit",
            remove_unused_columns=False,
            beta=beta,  # DPO-specific parameter
            max_length=max_length,
            max_prompt_length=max_prompt_length,
        )

        self.trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=dpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )

        logger.info("Starting DPO training...")
        train_result = self.trainer.train()

        self.training_history = {
            "train_loss": train_result.training_loss,
            "training_complete": True,
        }

        logger.info(f"DPO training complete. Model saved to {self.output_dir}")
        return self.training_history

    def save_model(self, save_dir: Optional[str] = None) -> None:
        """Save trained model"""
        if save_dir is None:
            save_dir = self.output_dir

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.trainer.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        logger.info(f"Model saved to {save_dir}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate text from prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def save_metrics(self, filepath: str) -> None:
        """Save training metrics"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Metrics saved to {filepath}")
