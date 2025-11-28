"""
PPO (Proximal Policy Optimization) Trainer
Implements PPO for LLM alignment with sparse and dense rewards
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)
from trl import PPOTrainer, PPOConfig

logger = logging.getLogger(__name__)


class ValueHead(nn.Module):
    """Value head for estimating expected returns"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            values: [batch_size] scalar value estimates
        """
        # Take last token's hidden state
        last_hidden = hidden_states[:, -1, :]  # [batch, hidden_size]
        values = self.linear(last_hidden).squeeze(-1)  # [batch]
        return values


class PPOAlignmentTrainer:
    """PPO trainer for LLM alignment with value function and KL penalty"""

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolLM2-135M-SFT-only",
        ref_model_id: Optional[str] = None,
        reward_model_path: str = "checkpoints/reward_model",
        output_dir: str = "checkpoints/ppo",
        load_in_8bit: bool = True,
        use_lora: bool = True,
        reward_type: str = "sparse",
    ):
        """
        Initialize PPO Trainer

        Args:
            model_id: Base model identifier
            ref_model_id: Reference model ID
            reward_model_path: Path to trained reward model
            output_dir: Output directory
            load_in_8bit: Use 8-bit quantization
            use_lora: Use LoRA adapters
            reward_type: "sparse" or "dense"
        """
        self.model_id = model_id
        self.ref_model_id = ref_model_id or model_id
        self.reward_model_path = reward_model_path
        self.output_dir = output_dir
        self.load_in_8bit = load_in_8bit
        self.use_lora = use_lora
        self.reward_type = reward_type

        self.model = None
        self.ref_model = None
        self.reward_model = None
        self.value_model = None  # Separate value function/critic
        self.value_head = None  # Value head for scalar value prediction
        self.tokenizer = None
        self.ppo_trainer = None
        self.training_history = {}

    def load_models(self) -> None:
        """Load policy, reference, reward, and value models"""
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

        # Value model: reuse policy model (same as policy)
        logger.info("Reusing policy model for value estimation...")
        self.value_model = self.model

        # Initialize value head
        hidden_size = self.value_model.config.hidden_size
        self.value_head = ValueHead(hidden_size)
        self.value_head.to(self.value_model.device)
        logger.info(f"Value head initialized with hidden size {hidden_size}")

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

    def get_rewards_batch(self, prompts: List[str], responses: List[str]) -> List[float]:
        """Get rewards for batch of prompt-response pairs"""
        rewards = []
        for prompt, response in zip(prompts, responses):
            reward = self.get_reward(prompt, response)
            rewards.append(reward)
        return rewards

    def compute_kl_penalty(
        self,
        prompt_logits: torch.Tensor,
        response_logits: torch.Tensor,
        ref_response_logits: torch.Tensor,
        kl_coeff: float = 0.05,
    ) -> torch.Tensor:
        """
        Compute KL penalty between policy and reference policy

        Args:
            prompt_logits: Prompt logits
            response_logits: Response logits from policy
            ref_response_logits: Response logits from reference
            kl_coeff: KL coefficient

        Returns:
            KL penalty
        """
        # Log probabilities
        log_probs = torch.nn.functional.log_softmax(response_logits, dim=-1)
        ref_log_probs = torch.nn.functional.log_softmax(ref_response_logits, dim=-1)

        # KL divergence
        ref_probs = torch.softmax(ref_response_logits, dim=-1)
        kl = torch.sum(ref_probs * (ref_log_probs - log_probs), dim=-1)

        return kl_coeff * kl

    def compute_advantages(
        self,
        rewards: List[float],
        values: List[float],
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> List[float]:
        """
        Compute Generalized Advantage Estimation (GAE)

        Args:
            rewards: List of rewards
            values: List of value estimates
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            List of advantages
        """
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages.insert(0, gae)

        return advantages

    def estimate_value(self, prompt: str, response: str) -> float:
        """
        Estimate value (expected return) for a response using value head

        Args:
            prompt: Input prompt
            response: Generated response

        Returns:
            Value estimate (scalar)
        """
        text = prompt + "\n" + response
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.value_model.device)

        with torch.no_grad():
            outputs = self.value_model(**inputs, output_hidden_states=True)
            # Use value head to predict scalar value
            hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
            value = self.value_head(hidden_states).item()  # [batch] -> scalar

        return value

    def compute_log_prob_ratio(
        self,
        prompt: str,
        response: str,
        old_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability ratio: log(π_θ(y|x) / π_θ_old(y|x))

        Args:
            prompt: Input prompt
            response: Generated response
            old_log_probs: Log probabilities from old policy

        Returns:
            Log probability ratio
        """
        text = prompt + "\n" + response
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        outputs = self.model(**inputs, output_hidden_states=False)
        new_log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)

        log_ratio = new_log_probs.sum() - old_log_probs.sum()
        return log_ratio

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        num_ppo_epochs: int = 4,
        per_device_train_batch_size: int = 4,
        per_device_eval_batch_size: int = 8,
        learning_rate: float = 1e-5,
        value_learning_rate: float = 1e-5,
        kl_coeff: float = 0.05,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        max_length: int = 512,
        max_prompt_length: int = 256,
        warmup_steps: int = 100,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 50,
        num_train_epochs: int = 3,
    ) -> Dict:
        """
        Train with PPO using proper advantage estimation and KL penalty

        LPPO(θ) = E(x,y)~D [min(r_θ*A, clip(r_θ, 1-ε, 1+ε)*A)] - β*KL(π_θ || π_ref)
        """
        if self.model is None:
            self.load_models()

        # Ensure models are in correct modes
        self.model.train()
        self.value_model.train()
        self.ref_model.eval()  # Reference model stays frozen
        self.reward_model.eval()  # Reward model stays frozen

        logger.info(f"Starting PPO training with {self.reward_type} rewards...")
        logger.info(f"KL coefficient: {kl_coeff}, Clip ratio: {clip_ratio}")

        # Verify trainable parameters
        policy_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        value_trainable = sum(p.numel() for p in self.value_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        logger.info(f"Policy trainable params: {policy_trainable:,} / {total_params:,}")
        logger.info(f"Value trainable params: {value_trainable:,} / {total_params:,}")

        if policy_trainable == 0 or value_trainable == 0:
            raise RuntimeError("No trainable parameters! LoRA might not be applied correctly.")

        # Setup separate optimizers
        policy_optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
        )
        # Value optimizer includes both value_model and value_head parameters
        value_params = list(self.value_model.parameters()) + list(self.value_head.parameters())
        value_optimizer = torch.optim.AdamW(
            [p for p in value_params if p.requires_grad],
            lr=value_learning_rate,
        )

        logger.info("Starting PPO training loop...")

        # Training loop
        metrics_history = []
        global_step = 0

        for epoch in range(num_train_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_train_epochs}")

            batch_count = 0
            epoch_rewards = []
            epoch_losses = []
            epoch_policy_losses = []
            epoch_value_losses = []
            epoch_kl_penalties = []

            # Iterate over dataset using indexing (FIXES CRITICAL BUG)
            for idx in range(0, len(train_dataset), per_device_train_batch_size):
                batch_count += 1

                # Get batch (FIXES CRITICAL BUG #1)
                batch_end = min(idx + per_device_train_batch_size, len(train_dataset))
                batch_data = train_dataset[idx:batch_end]

                prompts = batch_data["prompt"]
                if not isinstance(prompts, list):
                    prompts = [prompts]

                batch_responses = []
                batch_rewards = []
                batch_values = []
                batch_advantages = []
                batch_old_log_probs = []  # Store old log probs for PPO ratio

                # Generate responses and compute rewards for each prompt
                logger.info(f"Processing {len(prompts)} prompts in batch {batch_count}...")
                for idx, prompt in enumerate(prompts):
                    logger.info(f"  Generating response {idx+1}/{len(prompts)}...")
                    # Generate response
                    prompt_inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_prompt_length
                    ).to(self.model.device)

                    with torch.no_grad():
                        outputs = self.model.generate(
                            **prompt_inputs,
                            max_new_tokens=128,
                            do_sample=True,
                            top_p=0.9,
                        )
                        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        response_text = response_text.replace(prompt, "").strip()

                        # Compute old log probs for PPO ratio
                        full_text = prompt + "\n" + response_text
                        full_inputs = self.tokenizer(
                            full_text,
                            truncation=True,
                            max_length=max_length,
                            return_tensors="pt",
                            padding=True,
                        ).to(self.model.device)

                        # Get log probs from current policy (before update)
                        old_outputs = self.model(**full_inputs)
                        old_log_probs = torch.nn.functional.log_softmax(old_outputs.logits, dim=-1)
                        # Store mean log prob as a scalar for simplicity
                        old_log_prob = old_log_probs.mean().item()
                        batch_old_log_probs.append(old_log_prob)

                    # Compute reward
                    logger.info(f"  Computing reward for response {idx+1}...")
                    reward = self.get_reward(prompt, response_text)
                    batch_rewards.append(reward)
                    batch_responses.append(response_text)

                    # Compute value estimate
                    logger.info(f"  Computing value estimate for response {idx+1}...")
                    value = self.estimate_value(prompt, response_text)
                    batch_values.append(value)
                    logger.info(f"  Response {idx+1} done - Reward: {reward:.4f}, Value: {value:.4f}")

                    epoch_rewards.append(reward)

                # Compute advantages (FIXES CRITICAL BUG #3)
                batch_advantages = self.compute_advantages(
                    batch_rewards,
                    batch_values,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                )
                batch_advantages = np.array(batch_advantages)
                batch_advantages = (batch_advantages - np.mean(batch_advantages)) / (np.std(batch_advantages) + 1e-8)
                batch_advantages = batch_advantages.tolist()

                # Policy update with proper PPO clipped objective
                logger.info(f"Computing policy loss and updating...")
                policy_optimizer.zero_grad()
                policy_loss_total = 0

                for prompt, response, advantage, old_log_prob in zip(
                    prompts, batch_responses, batch_advantages, batch_old_log_probs
                ):
                    # Tokenize
                    full_text = prompt + "\n" + response
                    inputs = self.tokenizer(
                        full_text,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                        padding=True,
                    ).to(self.model.device)

                    # Forward pass through policy
                    outputs = self.model(**inputs)
                    new_logits = outputs.logits

                    # Compute new log probs
                    new_log_probs = torch.nn.functional.log_softmax(new_logits, dim=-1)
                    new_log_prob = new_log_probs.mean()  # Simplified: use mean

                    # Compute ratio: π_θ(y|x) / π_θ_old(y|x)
                    log_ratio = new_log_prob - old_log_prob
                    ratio = torch.exp(log_ratio)

                    # PPO clipped objective
                    advantage_tensor = torch.tensor(advantage, device=ratio.device)
                    policy_loss_unclipped = ratio * advantage_tensor
                    policy_loss_clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage_tensor
                    policy_loss = -torch.min(policy_loss_unclipped, policy_loss_clipped)

                    # Compute KL divergence with reference model
                    with torch.no_grad():
                        ref_outputs = self.ref_model(**inputs)
                        ref_logits = ref_outputs.logits

                    # KL divergence: sum_i ref_probs(i) * (log ref_probs(i) - log policy_probs(i))
                    log_policy_probs = torch.nn.functional.log_softmax(new_logits, dim=-1)
                    log_ref_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
                    ref_probs = torch.softmax(ref_logits, dim=-1)

                    kl_div = torch.sum(ref_probs * (log_ref_probs - log_policy_probs), dim=-1)
                    kl_penalty = kl_coeff * kl_div.mean()

                    epoch_kl_penalties.append(kl_penalty.item())

                    # Total loss: PPO + KL penalty
                    total_loss = policy_loss + kl_penalty
                    policy_loss_total += total_loss
                    epoch_policy_losses.append(total_loss.item())

                # Backward through all samples
                if policy_loss_total > 0:
                    (policy_loss_total / len(prompts)).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    policy_optimizer.step()

                # Value function update
                value_optimizer.zero_grad()
                value_loss_total = 0

                for prompt, response, reward in zip(prompts, batch_responses, batch_rewards):
                    full_text = prompt + "\n" + response
                    inputs = self.tokenizer(
                        full_text,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                        padding=True,
                    ).to(self.value_model.device)

                    # Get value prediction from value head
                    value_outputs = self.value_model(**inputs, output_hidden_states=True)
                    hidden_states = value_outputs.hidden_states[-1]  # [batch, seq_len, hidden]
                    predicted_value = self.value_head(hidden_states)  # [batch]

                    # Target is the actual reward (return)
                    target_value = torch.tensor([reward], device=predicted_value.device, dtype=predicted_value.dtype)

                    # MSE loss between predicted and actual returns
                    value_loss = torch.nn.functional.mse_loss(predicted_value, target_value)
                    value_loss_total += value_loss
                    epoch_value_losses.append(value_loss.item())

                if value_loss_total > 0:
                    (value_loss_total / len(prompts)).backward()
                    torch.nn.utils.clip_grad_norm_(value_params, max_norm=1.0)
                    value_optimizer.step()

                global_step += 1

                epoch_losses.append(float(np.mean(epoch_policy_losses[-len(prompts):])))

                # Log every N batches
                if batch_count % logging_steps == 0:
                    avg_reward = np.mean(epoch_rewards[-logging_steps * per_device_train_batch_size:])
                    avg_loss = np.mean(epoch_losses[-logging_steps:]) if epoch_losses else 0
                    avg_kl = np.mean(epoch_kl_penalties[-logging_steps * per_device_train_batch_size:]) if epoch_kl_penalties else 0
                    logger.info(
                        f"Batch {batch_count}: Loss = {avg_loss:.4f}, Reward = {avg_reward:.4f}, KL = {avg_kl:.4f}"
                    )

            epoch_metrics = {
                "epoch": epoch + 1,
                "mean_loss": float(np.mean(epoch_losses)) if epoch_losses else 0,
                "mean_policy_loss": float(np.mean(epoch_policy_losses)) if epoch_policy_losses else 0,
                "mean_value_loss": float(np.mean(epoch_value_losses)) if epoch_value_losses else 0,
                "mean_kl_penalty": float(np.mean(epoch_kl_penalties)) if epoch_kl_penalties else 0,
                "mean_reward": float(np.mean(epoch_rewards)) if epoch_rewards else 0,
                "std_reward": float(np.std(epoch_rewards)) if epoch_rewards else 0,
                "max_reward": float(np.max(epoch_rewards)) if epoch_rewards else 0,
                "min_reward": float(np.min(epoch_rewards)) if epoch_rewards else 0,
            }
            metrics_history.append(epoch_metrics)
            logger.info(f"Epoch {epoch + 1} metrics: {epoch_metrics}")

            # Save checkpoint
            save_frequency = max(1, save_steps // 1000) if save_steps >= 1000 else 1
            if (epoch + 1) % save_frequency == 0 or epoch == num_train_epochs - 1:
                save_dir = f"{self.output_dir}/checkpoint-epoch-{epoch+1}"
                self.save_model(save_dir)

        self.training_history = {
            "metrics": metrics_history,
            "training_complete": True,
        }

        logger.info("PPO training complete")
        return self.training_history

    def save_model(self, save_dir: Optional[str] = None) -> None:
        """Save trained policy and value models"""
        if save_dir is None:
            save_dir = self.output_dir

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Save policy model
        self.model.save_pretrained(f"{save_dir}/policy")

        # Save value model
        self.value_model.save_pretrained(f"{save_dir}/value")

        # Save value head
        torch.save(self.value_head.state_dict(), f"{save_dir}/value_head.pt")

        # Save tokenizer
        self.tokenizer.save_pretrained(save_dir)

        logger.info(f"Models saved to {save_dir} (policy, value, value_head)")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate text from prompt"""
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,  # Changed
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