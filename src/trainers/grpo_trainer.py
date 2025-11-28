"""
GRPO (Group Relative Policy Optimization) Trainer
Implements GRPO for efficient LLM alignment without separate value function
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

logger = logging.getLogger(__name__)


class GRPOAlignmentTrainer:
    """GRPO trainer for efficient LLM alignment"""

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolLM2-135M-SFT-only",
        ref_model_id: Optional[str] = None,
        reward_model_path: str = "checkpoints/reward_model",
        output_dir: str = "checkpoints/grpo",
        load_in_8bit: bool = True,
        use_lora: bool = True,
        group_size: int = 4,
    ):
        """
        Initialize GRPO Trainer

        Args:
            model_id: Base model identifier
            ref_model_id: Reference model ID
            reward_model_path: Path to trained reward model
            output_dir: Output directory
            load_in_8bit: Use 8-bit quantization
            use_lora: Use LoRA adapters
            group_size: Number of responses per prompt
        """
        self.model_id = model_id
        self.ref_model_id = ref_model_id or model_id
        self.reward_model_path = reward_model_path
        self.output_dir = output_dir
        self.load_in_8bit = load_in_8bit
        self.use_lora = use_lora
        self.group_size = group_size

        self.model = None
        self.ref_model = None
        self.reward_model = None
        self.tokenizer = None
        self.training_history = {}

    def load_models(self) -> None:
        """Load policy, reference, and reward models"""
        logger.info(f"Loading policy model: {self.model_id}")
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

        # Apply LoRA if requested
        if self.use_lora:
            logger.info("Applying LoRA adapters...")
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)

        # Reference model: use deepcopy of policy model (frozen) to save memory
        logger.info("Creating reference model as frozen copy of policy...")
        import copy
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()  # Freeze reference model

        # Load reward model
        logger.info(f"Loading reward model from {self.reward_model_path}")
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            self.reward_model_path,
            device_map="auto",
        )
        self.reward_model.eval()

        logger.info("All models loaded successfully")

    def get_reward(self, prompt: str, response: str) -> float:
        """Get reward from reward model"""
        text = prompt + "\n" + response
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.reward_model.device)

        with torch.no_grad():
            outputs = self.reward_model(**inputs)
            logits = outputs.logits

            # Handle both regression (num_labels=1) and binary classification (num_labels=2)
            if logits.shape[-1] == 1:
                # Regression model (e.g., OpenAssistant): single scalar output
                reward = logits[0, 0].item()
            else:
                # Binary classification: use logit for "chosen" class (label 1)
                reward = logits[0, 1].item()

        return reward

    def compute_kl_penalty(self, prompt: str, response: str) -> float:
        """Compute KL divergence between policy and reference model"""
        full_text = prompt + "\n" + response
        inputs = self.tokenizer(
            full_text,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            # Get log probs from policy model
            policy_outputs = self.model(**inputs, output_hidden_states=False)
            policy_logits = policy_outputs.logits
            policy_log_probs = torch.nn.functional.log_softmax(policy_logits, dim=-1)

            # Get log probs from reference model
            ref_outputs = self.ref_model(**inputs, output_hidden_states=False)
            ref_logits = ref_outputs.logits
            ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)

            # Compute KL divergence: KL(policy || ref) = sum(policy * (log(policy) - log(ref)))
            # For numerical stability, we use the logits directly
            kl_div = torch.nn.functional.kl_div(
                ref_log_probs,
                policy_log_probs,
                log_target=True,
                reduction='batchmean'
            )

        return kl_div.item()

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate a single response"""
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
        # Remove prompt from response
        response = response.replace(prompt, "").strip()
        return response

    def compute_group_relative_advantage(
        self,
        rewards: List[float],
    ) -> List[float]:
        """
        Compute group-relative advantage

        As per GRPO paper:
        Â_i = (R_i - R̄) / σ_R

        Args:
            rewards: List of rewards for group

        Returns:
            List of normalized advantages
        """
        rewards_array = np.array(rewards)
        mean_reward = np.mean(rewards_array)
        std_reward = np.std(rewards_array)

        if std_reward == 0:
            std_reward = 1e-8

        advantages = (rewards_array - mean_reward) / std_reward
        return advantages.tolist()

    def train(
        self,
        train_dataset: Dataset,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        learning_rate: float = 5e-5,
        kl_coeff: float = 0.05,
        max_length: int = 512,
        max_prompt_length: int = 256,
        warmup_steps: int = 100,
        save_steps: int = 500,
        logging_steps: int = 50,
    ) -> Dict:
        """
        Train with GRPO

        Args:
            train_dataset: Training dataset
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size
            learning_rate: Learning rate
            kl_coeff: KL coefficient
            max_length: Maximum sequence length
            max_prompt_length: Maximum prompt length
            warmup_steps: Warmup steps
            save_steps: Save checkpoint frequency
            logging_steps: Logging frequency

        Returns:
            Training metrics dictionary
        """
        if self.model is None:
            self.load_models()

        logger.info(f"Starting GRPO training with group size {self.group_size}...")

        # Verify trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_count = sum(p.numel() for p in trainable_params)
        if not trainable_params:
            raise RuntimeError("No trainable parameters found! LoRA may not be applied correctly.")
        logger.info(f"Trainable params: {trainable_count:,} / {total_params:,} ({100 * trainable_count / total_params:.2f}%)")

        # Setup optimizer (only trainable parameters)
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
        )

        metrics_history = []
        global_step = 0

        for epoch in range(num_train_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_train_epochs}")

            epoch_losses = []
            epoch_rewards = []
            epoch_kl_penalties = []
            batch_count = 0

            for idx in range(0, len(train_dataset), per_device_train_batch_size):
                batch_count += 1
                batch_data = train_dataset[idx:idx + per_device_train_batch_size]

                batch_rewards = []
                batch_responses = []
                batch_prompts = []
                batch_old_log_probs = []
                batch_kl_penalties = []

                # For each prompt, sample group_size responses
                logger.info(f"Batch {batch_count}: Processing {len(batch_data['prompt'])} prompts...")
                for prompt_idx, prompt in enumerate(batch_data["prompt"]):
                    logger.info(f"  Prompt {prompt_idx+1}/{len(batch_data['prompt'])}: Generating {self.group_size} responses...")
                    group_rewards = []
                    group_responses = []
                    group_old_log_probs = []
                    group_kl_penalties = []

                    # Generate group_size responses
                    for resp_idx in range(self.group_size):
                        logger.info(f"    Generating response {resp_idx+1}/{self.group_size}...")
                        response = self.generate_response(prompt)

                        # Get reward from reward model
                        logger.info(f"    Computing reward...")
                        rm_reward = self.get_reward(prompt, response)

                        # Compute KL penalty
                        logger.info(f"    Computing KL penalty...")
                        kl_penalty = self.compute_kl_penalty(prompt, response)

                        # Final reward = reward_model_score - kl_coeff * KL(policy || ref)
                        reward = rm_reward - kl_coeff * kl_penalty
                        logger.info(f"    Response {resp_idx+1} done - RM Reward: {rm_reward:.4f}, KL: {kl_penalty:.6f}, Final: {reward:.4f}")

                        # Store old log prob (for PPO clipping)
                        full_text = prompt + "\n" + response
                        inputs = self.tokenizer(
                            full_text,
                            truncation=True,
                            max_length=max_length,
                            return_tensors="pt",
                        ).to(self.model.device)

                        with torch.no_grad():
                            old_outputs = self.model(**inputs)
                            old_logits = old_outputs.logits
                            old_log_probs = torch.nn.functional.log_softmax(old_logits, dim=-1)
                            # Average log prob across tokens as simple proxy
                            old_log_prob = old_log_probs.mean().item()

                        group_responses.append(response)
                        group_rewards.append(reward)
                        group_old_log_probs.append(old_log_prob)
                        group_kl_penalties.append(kl_penalty)

                    batch_prompts.append(prompt)
                    batch_responses.append(group_responses)
                    batch_rewards.append(group_rewards)
                    batch_old_log_probs.append(group_old_log_probs)
                    batch_kl_penalties.extend(group_kl_penalties)
                    epoch_rewards.extend(group_rewards)
                    epoch_kl_penalties.extend(group_kl_penalties)

                # Compute loss and update
                logger.info(f"Batch {batch_count}: Computing policy loss and updating weights...")
                optimizer.zero_grad()

                # Compute loss for each prompt group
                for prompt, responses, rewards, old_log_probs in zip(
                    batch_prompts, batch_responses, batch_rewards, batch_old_log_probs
                ):
                    # Compute group-relative advantages
                    advantages = self.compute_group_relative_advantage(rewards)

                    for response, advantage, old_log_prob in zip(responses, advantages, old_log_probs):
                        # Combine prompt and response for training
                        full_text = prompt + "\n" + response
                        inputs = self.tokenizer(
                            full_text,
                            truncation=True,
                            max_length=max_length,
                            return_tensors="pt",
                        ).to(self.model.device)

                        if len(inputs.input_ids[0]) < 2:
                            continue

                        # Get current log probs
                        outputs = self.model(**inputs)
                        logits = outputs.logits
                        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                        new_log_prob = log_probs.mean()

                        # Compute PPO clipped objective
                        # ratio = exp(log π_θ(a|s) - log π_θ_old(a|s))
                        log_ratio = new_log_prob - old_log_prob
                        ratio = torch.exp(log_ratio)

                        # Clipped surrogate objective
                        clip_ratio = 0.2  # Standard PPO clip value
                        advantage_tensor = torch.tensor(advantage, device=self.model.device)

                        surr1 = ratio * advantage_tensor
                        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage_tensor

                        # Loss = -min(surr1, surr2) (negative because we want gradient ascent)
                        policy_loss = -torch.min(surr1, surr2)

                        policy_loss.backward()
                        epoch_losses.append(policy_loss.item())

                optimizer.step()  # Update weights after all responses in batch
                global_step += 1
                logger.info(f"Batch {batch_count} complete!")

                if batch_count % logging_steps == 0:
                    avg_loss = np.mean(epoch_losses[-logging_steps:]) if epoch_losses else 0
                    avg_reward = np.mean(epoch_rewards[-self.group_size * logging_steps:]) if epoch_rewards else 0
                    avg_kl = np.mean(epoch_kl_penalties[-self.group_size * logging_steps:]) if epoch_kl_penalties else 0
                    logger.info(f"Batch {batch_count}: Loss = {avg_loss:.4f}, Reward = {avg_reward:.4f}, KL = {avg_kl:.6f}")

            epoch_metrics = {
                "epoch": epoch + 1,
                "mean_loss": float(np.mean(epoch_losses)) if epoch_losses else 0,
                "mean_reward": float(np.mean(epoch_rewards)) if epoch_rewards else 0,
                "std_reward": float(np.std(epoch_rewards)) if epoch_rewards else 0,
                "max_reward": float(np.max(epoch_rewards)) if epoch_rewards else 0,
                "min_reward": float(np.min(epoch_rewards)) if epoch_rewards else 0,
                "mean_kl_penalty": float(np.mean(epoch_kl_penalties)) if epoch_kl_penalties else 0,
                "std_kl_penalty": float(np.std(epoch_kl_penalties)) if epoch_kl_penalties else 0,
            }
            metrics_history.append(epoch_metrics)
            logger.info(f"Epoch {epoch + 1} metrics: {epoch_metrics}")

        # Save model only at the end
        logger.info("Saving final GRPO model...")
        self.save_model(self.output_dir)

        self.training_history = {
            "metrics": metrics_history,
            "training_complete": True,
        }

        logger.info("GRPO training complete")
        return self.training_history

    def save_model(self, save_dir: Optional[str] = None) -> None:
        """Save trained model"""
        if save_dir is None:
            save_dir = self.output_dir

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_dir)
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
