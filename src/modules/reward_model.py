"""
Reward Model training for PPO alignment
Trains a classifier to predict human preferences (chosen vs rejected)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


class RewardModelTrainer:
    """Train a reward model from preference data"""

    def __init__(
        self,
        model_id: str = "distilbert-base-uncased",
        output_dir: str = "checkpoints/reward_model",
        load_in_8bit: bool = True,
    ):
        """
        Initialize Reward Model Trainer

        Args:
            model_id: HuggingFace model identifier
            output_dir: Directory to save checkpoints
            load_in_8bit: Use 8-bit quantization
        """
        self.model_id = model_id
        self.output_dir = output_dir
        self.load_in_8bit = load_in_8bit
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.metrics_history = {}

    # def load_model(self) -> None:
    #     """Load pre-trained model for sequence classification"""
    #     logger.info(f"Loading reward model: {self.model_id}")

    #     self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
    #     if self.tokenizer.pad_token is None:
    #         self.tokenizer.pad_token = self.tokenizer.eos_token

    #     model_kwargs = {
    #         "num_labels": 2,  # Binary classification: chosen (1) vs rejected (0)
    #         "problem_type": "single_label_classification",
    #     }

    #     if self.load_in_8bit:
    #         from transformers import BitsAndBytesConfig
    #         bnb_config = BitsAndBytesConfig(
    #             load_in_8bit=True,
    #             bnb_8bit_compute_dtype=torch.float16,
    #         )
    #         self.model = AutoModelForSequenceClassification.from_pretrained(
    #             self.model_id,
    #             quantization_config=bnb_config,
    #             device_map="auto",
    #             **model_kwargs
    #         )
    #     else:
    #         self.model = AutoModelForSequenceClassification.from_pretrained(
    #             self.model_id,
    #             **model_kwargs
    #         )

    #     logger.info("Model loaded successfully")

    def load_model(self) -> None:
        """Load pre-trained model for sequence classification"""
        logger.info(f"Loading reward model: {self.model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # OpenAssistant model is regression (num_labels=1), don't override
        # For training from scratch, we'd use binary classification (num_labels=2)
        model_kwargs = {}
        if "OpenAssistant" not in self.model_id:
            model_kwargs = {
                "num_labels": 2,
                "problem_type": "single_label_classification",
            }

        # 8-bit quantization only works on CUDA
        use_8bit = self.load_in_8bit and torch.cuda.is_available()

        if use_8bit:
            from transformers import BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model

            logger.info("Using 8-bit quantization with LoRA")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto",
                **model_kwargs
            )

            # Add LoRA adapters for fine-tuning quantized model
            # DeBERTa uses different layer names than GPT models
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["query_proj", "value_proj"],  # DeBERTa attention layers
                lora_dropout=0.05,
                bias="none",
                task_type="SEQ_CLS"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        else:
            if self.load_in_8bit and not torch.cuda.is_available():
                logger.warning("8-bit quantization requested but CUDA not available. Loading full model on CPU.")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                **model_kwargs
            )

        # Store actual quantization state
        self.load_in_8bit = use_8bit
        logger.info("Model loaded successfully")


    def preprocess_function(self, examples: Dict) -> Dict:
        """Tokenize examples for reward model"""
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        # For regression models, convert labels to float
        # For classification, keep as int
        if "OpenAssistant" in self.model_id:
            tokenized["labels"] = [float(label) for label in examples["label"]]
        else:
            tokenized["labels"] = examples["label"]
        return tokenized

    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute metrics during training"""
        predictions, labels = eval_pred

        # Handle both regression and classification
        if predictions.shape[-1] == 1 or len(predictions.shape) == 1:
            # Regression: compute accuracy based on whether prediction > 0.5
            if len(predictions.shape) == 2:
                predictions = predictions[:, 0]
            accuracy = ((predictions > 0.5) == labels).mean()
        else:
            # Classification: use argmax
            predictions = np.argmax(predictions, axis=1)
            accuracy = (predictions == labels).mean()

        return {"accuracy": accuracy}

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        num_train_epochs: int = 2,
        per_device_train_batch_size: int = 16,
        per_device_eval_batch_size: int = 32,
        learning_rate: float = 1e-4,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 50,
    ) -> Dict[str, float]:
        """
        Train the reward model

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device
            per_device_eval_batch_size: Eval batch size
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay for optimizer
            save_steps: Steps between checkpoints
            eval_steps: Steps between evaluations
            logging_steps: Steps between logging

        Returns:
            Dictionary with training results
        """
        if self.model is None:
            self.load_model()

        logger.info("Preprocessing datasets...")
        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=["text", "prompt", "response"],
            desc="Processing train data"
        )
        eval_dataset = eval_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=["text", "prompt", "response"],
            desc="Processing eval data"
        )

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            save_steps=save_steps,
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            bf16=True,  # BF16 for better compatibility (avoids FP16 gradient unscaling)
            fp16=False,  # Avoid FP16 unscaling issues
            remove_unused_columns=True,
            optim="paged_adamw_32bit",
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
        )

        logger.info("Starting reward model training...")
        train_result = self.trainer.train()

        self.metrics_history = {
            "train_loss": train_result.training_loss,
            "eval_metrics": self.trainer.evaluate(),
        }

        logger.info(f"Training complete. Best model saved to {self.output_dir}")
        return self.metrics_history

    def get_reward(self, prompt: str, response: str) -> float:
        """
        Get reward score for a prompt-response pair

        Args:
            prompt: Input prompt
            response: Generated response

        Returns:
            Reward score (scalar)
        """
        if self.model is None:
            self.load_model()

        text = prompt + "\n" + response
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Handle both regression (num_labels=1) and binary classification (num_labels=2)
            if logits.shape[-1] == 1:
                # Regression model (e.g., OpenAssistant): single scalar output
                reward = logits[0, 0].item()
            else:
                # Binary classification: use logit for "chosen" class (label 1)
                reward = logits[0, 1].item()

        return reward

    def get_rewards_batch(self, prompts: list, responses: list) -> list:
        """
        Get rewards for a batch of prompt-response pairs

        Args:
            prompts: List of prompts
            responses: List of responses

        Returns:
            List of reward scores
        """
        rewards = []
        for prompt, response in zip(prompts, responses):
            reward = self.get_reward(prompt, response)
            rewards.append(reward)
        return rewards

    def save_model(self, save_dir: str) -> None:
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train first.")

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        logger.info(f"Model saved to {save_dir}")

    def load_trained_model(self, model_dir: str) -> None:
        """Load a previously trained model"""
        logger.info(f"Loading trained model from {model_dir}")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        logger.info("Model loaded successfully")

    def save_metrics(self, filepath: str) -> None:
        """Save training metrics to JSON"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy values to Python types for JSON serialization
        metrics = {
            k: float(v) if isinstance(v, (np.floating, float)) else v
            for k, v in self.metrics_history.items()
        }
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {filepath}")
