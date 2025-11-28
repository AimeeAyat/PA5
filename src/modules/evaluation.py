"""
Evaluation metrics for LLM alignment
Measures catastrophic forgetting, verbosity bias, and reward hacking
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Compute evaluation metrics for aligned models"""

    def __init__(
        self,
        ref_model_id: str,
        device: str = "cuda",
    ):
        """
        Initialize evaluation metrics

        Args:
            ref_model_id: Reference (base) model identifier
            device: Device to use for computation
        """
        self.ref_model_id = ref_model_id
        self.device = device
        self.ref_model = None
        self.ref_tokenizer = None

    def load_reference_model(self) -> None:
        """Load reference model for KL divergence computation"""
        logger.info(f"Loading reference model: {self.ref_model_id}")
        self.ref_tokenizer = AutoTokenizer.from_pretrained(self.ref_model_id)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.ref_model_id,
            device_map=self.device,
            load_in_8bit=True,
        )
        logger.info("Reference model loaded")

    # ========================================================================
    # CATASTROPHIC FORGETTING METRICS
    # ========================================================================

    def compute_kl_divergence(
        self,
        aligned_model,
        aligned_tokenizer,
        prompts: List[str],
        max_length: int = 512,
    ) -> float:
        """
        Compute KL divergence between aligned and reference models

        Args:
            aligned_model: Aligned model
            aligned_tokenizer: Aligned model tokenizer
            prompts: List of prompts
            max_length: Maximum sequence length

        Returns:
            Average KL divergence
        """
        if self.ref_model is None:
            self.load_reference_model()

        kl_divergences = []

        for prompt in prompts[:10]:  # Limit to 10 for computational efficiency
            inputs = self.ref_tokenizer(
                prompt,
                truncation=True,
                max_length=max_length // 2,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                # Reference model logits
                ref_outputs = self.ref_model(**inputs, output_hidden_states=False)
                ref_logits = ref_outputs.logits[:, -1, :]
                ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)

                # Aligned model logits
                aligned_inputs = aligned_tokenizer(
                    prompt,
                    truncation=True,
                    max_length=max_length // 2,
                    return_tensors="pt",
                ).to(self.device)
                aligned_outputs = aligned_model(**aligned_inputs)
                aligned_logits = aligned_outputs.logits[:, -1, :]
                aligned_log_probs = torch.nn.functional.log_softmax(aligned_logits, dim=-1)

                # KL divergence: sum_i p_ref(i) * (log p_ref(i) - log p_aligned(i))
                ref_probs = torch.softmax(ref_logits, dim=-1)
                kl = torch.sum(ref_probs * (ref_log_probs - aligned_log_probs), dim=-1)
                kl_divergences.append(kl.item())

        avg_kl = np.mean(kl_divergences)
        return avg_kl

    def compute_perplexity(
        self,
        model,
        tokenizer,
        texts: List[str],
        max_length: int = 512,
    ) -> float:
        """
        Compute perplexity on a text corpus

        Args:
            model: Model to evaluate
            tokenizer: Model tokenizer
            texts: List of texts
            max_length: Maximum sequence length

        Returns:
            Perplexity score
        """
        model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for text in texts[:20]:  # Limit to 20 samples
                inputs = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)

                if len(inputs.input_ids[0]) < 2:
                    continue

                outputs = model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss
                total_loss += loss.item() * (len(inputs.input_ids[0]) - 1)
                total_tokens += len(inputs.input_ids[0]) - 1

        if total_tokens == 0:
            return float('inf')

        perplexity = np.exp(total_loss / total_tokens)
        return perplexity

    # ========================================================================
    # VERBOSITY BIAS METRICS
    # ========================================================================

    def compute_response_length_stats(
        self,
        responses: List[str],
    ) -> Dict[str, float]:
        """
        Compute length statistics for responses

        Args:
            responses: List of response strings

        Returns:
            Dictionary with length statistics
        """
        lengths = [len(r.split()) for r in responses]

        stats = {
            "mean_length": float(np.mean(lengths)),
            "median_length": float(np.median(lengths)),
            "std_length": float(np.std(lengths)),
            "min_length": float(np.min(lengths)),
            "max_length": float(np.max(lengths)),
        }

        return stats

    def compute_verbosity_metrics(
        self,
        responses: List[str],
        query_types: Optional[List[str]] = None,
    ) -> Dict[str, any]:
        """
        Compute comprehensive verbosity metrics

        Args:
            responses: List of responses
            query_types: Optional list of query type labels

        Returns:
            Dictionary with verbosity metrics
        """
        lengths = [len(r.split()) for r in responses]

        # Basic statistics
        stats = self.compute_response_length_stats(responses)

        # Skewness (right skew indicates occasional long responses)
        from scipy import stats as sp_stats
        skewness = float(sp_stats.skew(lengths))
        kurtosis = float(sp_stats.kurtosis(lengths))

        metrics = {
            **stats,
            "skewness": skewness,
            "kurtosis": kurtosis,
        }

        # Stratified by query type if provided
        if query_types is not None:
            unique_types = set(query_types)
            for q_type in unique_types:
                type_lengths = [
                    lengths[i] for i in range(len(query_types))
                    if query_types[i] == q_type
                ]
                if type_lengths:
                    metrics[f"{q_type}_mean_length"] = float(np.mean(type_lengths))
                    metrics[f"{q_type}_std_length"] = float(np.std(type_lengths))

        return metrics

    # ========================================================================
    # REWARD HACKING METRICS
    # ========================================================================

    def test_reward_sensitivity(
        self,
        reward_model,
        prompt: str,
        response: str,
        perturbations: List[str],
    ) -> Dict[str, float]:
        """
        Test reward model sensitivity to perturbations

        Args:
            reward_model: Trained reward model
            prompt: Input prompt
            response: Original response
            perturbations: List of perturbed versions of response

        Returns:
            Dictionary with reward scores for each perturbation
        """
        rewards = {
            "original": reward_model.get_reward(prompt, response),
        }

        for i, perturbed in enumerate(perturbations):
            rewards[f"perturbation_{i}"] = reward_model.get_reward(prompt, perturbed)

        return rewards

    def detect_reward_hacking(
        self,
        reward_model,
        prompts: List[str],
        good_responses: List[str],
        bad_responses: List[str],
    ) -> Dict[str, float]:
        """
        Detect reward hacking behavior

        Args:
            reward_model: Trained reward model
            prompts: List of prompts
            good_responses: Correct/good responses
            bad_responses: Incorrect but plausible responses

        Returns:
            Reward hacking metrics
        """
        good_rewards = []
        bad_rewards = []

        for prompt, good, bad in zip(prompts[:20], good_responses[:20], bad_responses[:20]):
            good_rewards.append(reward_model.get_reward(prompt, good))
            bad_rewards.append(reward_model.get_reward(prompt, bad))

        good_rewards = np.array(good_rewards)
        bad_rewards = np.array(bad_rewards)

        metrics = {
            "good_response_mean_reward": float(np.mean(good_rewards)),
            "good_response_std_reward": float(np.std(good_rewards)),
            "bad_response_mean_reward": float(np.mean(bad_rewards)),
            "bad_response_std_reward": float(np.std(bad_rewards)),
            "reward_gap": float(np.mean(good_rewards) - np.mean(bad_rewards)),
            "hackling_rate": float(np.mean(bad_rewards > good_rewards)),
        }

        return metrics

    def evaluate_all(
        self,
        aligned_model,
        aligned_tokenizer,
        reward_model,
        eval_prompts: List[str],
        eval_texts: List[str],
        eval_responses: List[str],
        query_types: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run all evaluation metrics

        Args:
            aligned_model: Model to evaluate
            aligned_tokenizer: Model tokenizer
            reward_model: Trained reward model for evaluation
            eval_prompts: List of prompts for KL/perplexity
            eval_texts: List of texts for perplexity
            eval_responses: List of generated responses
            query_types: Optional query type labels

        Returns:
            Dictionary with all metrics
        """
        logger.info("Computing evaluation metrics...")
        results = {}

        # Catastrophic forgetting
        logger.info("Computing KL divergence...")
        results["kl_divergence"] = self.compute_kl_divergence(
            aligned_model, aligned_tokenizer, eval_prompts
        )

        logger.info("Computing perplexity...")
        results["perplexity"] = self.compute_perplexity(
            aligned_model, aligned_tokenizer, eval_texts
        )

        # Verbosity bias
        logger.info("Computing verbosity metrics...")
        results["verbosity_metrics"] = self.compute_verbosity_metrics(
            eval_responses, query_types
        )

        return results

    def save_metrics(self, metrics: Dict, filepath: str) -> None:
        """Save metrics to JSON"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            return obj

        serializable_metrics = convert_to_serializable(metrics)

        with open(filepath, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)

        logger.info(f"Metrics saved to {filepath}")
