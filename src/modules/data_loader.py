"""
Data loading and preprocessing for Task 2: LLM Alignment
Loads Intel/orca_dpo_pairs dataset and prepares for alignment training
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import datasets
from datasets import Dataset, DatasetDict
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and preprocess data for alignment training"""

    def __init__(
        self,
        dataset_name: str = "Intel/orca_dpo_pairs",
        split: str = "train",
        max_samples: Optional[int] = None,
        eval_size: float = 0.1,
        seed: int = 42,
    ):
        """
        Initialize DataLoader

        Args:
            dataset_name: HuggingFace dataset identifier
            split: Dataset split to use
            max_samples: Limit number of samples
            eval_size: Fraction for evaluation split
            seed: Random seed for reproducibility
        """
        self.dataset_name = dataset_name
        self.split = split
        self.max_samples = max_samples
        self.eval_size = eval_size
        self.seed = seed
        self.raw_dataset = None
        self.processed_dataset = None

    def load_dataset(self) -> Dataset:
        """Load raw dataset from HuggingFace"""
        logger.info(f"Loading dataset: {self.dataset_name}")
        dataset = datasets.load_dataset(self.dataset_name, split=self.split)

        if self.max_samples:
            logger.info(f"Limiting to {self.max_samples} samples")
            dataset = dataset.select(range(min(self.max_samples, len(dataset))))

        self.raw_dataset = dataset
        logger.info(f"Loaded {len(dataset)} samples")
        return dataset

    def preprocess_for_dpo(self) -> Dataset:
        """
        Preprocess dataset for DPO training.
        Expected format: prompt, chosen (preferred), rejected (less-preferred)
        """
        if self.raw_dataset is None:
            self.load_dataset()

        logger.info("Preprocessing for DPO training...")

        def process_dpo(example):
            return {
                "prompt": example["question"],
                "chosen": example["chosen"],
                "rejected": example["rejected"],
            }

        processed = self.raw_dataset.map(
            process_dpo,
            remove_columns=self.raw_dataset.column_names,
            desc="Processing for DPO"
        )

        self.processed_dataset = processed
        return processed

    def preprocess_for_reward_model(self) -> Dataset:
        """
        Preprocess dataset for Reward Model training.
        Creates examples with label 1 for chosen and 0 for rejected.
        """
        if self.raw_dataset is None:
            self.load_dataset()

        logger.info("Preprocessing for Reward Model training...")

        data_for_rm = []

        for example in self.raw_dataset:
            # Positive example (chosen)
            data_for_rm.append({
                "text": example["question"] + "\n" + example["chosen"],
                "label": 1,
                "prompt": example["question"],
                "response": example["chosen"],
            })
            # Negative example (rejected)
            data_for_rm.append({
                "text": example["question"] + "\n" + example["rejected"],
                "label": 0,
                "prompt": example["question"],
                "response": example["rejected"],
            })

        rm_dataset = Dataset.from_dict({
            "text": [d["text"] for d in data_for_rm],
            "label": [d["label"] for d in data_for_rm],
            "prompt": [d["prompt"] for d in data_for_rm],
            "response": [d["response"] for d in data_for_rm],
        })

        self.processed_dataset = rm_dataset
        return rm_dataset

    def preprocess_for_ppo(self) -> Dataset:
        """
        Preprocess dataset for PPO training.
        For PPO, we need prompts for generation and reference responses.
        """
        if self.raw_dataset is None:
            self.load_dataset()

        logger.info("Preprocessing for PPO training...")

        def process_ppo(example):
            return {
                "prompt": example["question"],
                "reference_response": example["chosen"],
                "preferred_response": example["chosen"],
                "non_preferred_response": example["rejected"],
            }

        processed = self.raw_dataset.map(
            process_ppo,
            remove_columns=self.raw_dataset.column_names,
            desc="Processing for PPO"
        )

        self.processed_dataset = processed
        return processed

    def preprocess_for_grpo(self) -> Dataset:
        """
        Preprocess dataset for GRPO training.
        GRPO needs prompts and reference responses (for rewards).
        """
        if self.raw_dataset is None:
            self.load_dataset()

        logger.info("Preprocessing for GRPO training...")

        def process_grpo(example):
            return {
                "prompt": example["question"],
                "preferred_response": example["chosen"],
                "non_preferred_response": example["rejected"],
            }

        processed = self.raw_dataset.map(
            process_grpo,
            remove_columns=self.raw_dataset.column_names,
            desc="Processing for GRPO"
        )

        self.processed_dataset = processed
        return processed

    def train_test_split(
        self,
        dataset: Optional[Dataset] = None,
        test_size: float = 0.1,
    ) -> Tuple[Dataset, Dataset]:
        """Split dataset into train and test"""
        if dataset is None:
            dataset = self.processed_dataset

        if dataset is None:
            raise ValueError("No processed dataset available. Call preprocess first.")

        split_dataset = dataset.train_test_split(
            test_size=test_size,
            seed=self.seed
        )

        logger.info(
            f"Train/Test split: {len(split_dataset['train'])} / {len(split_dataset['test'])}"
        )

        return split_dataset["train"], split_dataset["test"]

    def get_prompts_for_evaluation(self, num_prompts: int = 50) -> List[str]:
        """Extract prompts for evaluation"""
        if self.raw_dataset is None:
            self.load_dataset()

        prompts = [example["question"] for example in self.raw_dataset.select(
            range(min(num_prompts, len(self.raw_dataset)))
        )]
        return prompts

    def save_dataset(self, filepath: str) -> None:
        """Save processed dataset to disk"""
        if self.processed_dataset is None:
            raise ValueError("No processed dataset. Call preprocess first.")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.processed_dataset.save_to_disk(filepath)
        logger.info(f"Dataset saved to {filepath}")

    def load_dataset_from_disk(self, filepath: str) -> Dataset:
        """Load dataset from disk"""
        dataset = Dataset.load_from_disk(filepath)
        self.processed_dataset = dataset
        logger.info(f"Dataset loaded from {filepath}")
        return dataset
