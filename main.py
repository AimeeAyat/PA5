"""
Main training script for Task 2: LLM Alignment
Implements and compares DPO, PPO, and GRPO alignment methods
"""

import json
import logging
from pathlib import Path

import torch
import numpy as np

# Import local modules
from config import (
    BASE_MODEL_ID, REFERENCE_MODEL_ID, DATASET_NAME, DATASET_SPLIT,
    MAX_SAMPLES, TEST_PROMPTS_NUM, RESULTS_DIR, CHECKPOINTS_DIR,
    VISUALIZATIONS_DIR, SAMPLES_DIR, DPOConfig, PPOConfig, PPORewardModelConfig, GRPOConfig,
)
from src.utils import setup_logger, ensure_dir, set_seed, setup_device
from src.modules.data_loader import DataLoader
from src.modules.reward_model import RewardModelTrainer
from src.modules.evaluation import EvaluationMetrics
from src.modules.visualization import ResultsVisualizer
from src.trainers import DPOAlignmentTrainer, PPOAlignmentTrainer, GRPOAlignmentTrainer

# Setup logging
ensure_dir(RESULTS_DIR)
logger = setup_logger(
    "Task2_LLM_Alignment",
    log_file=f"{RESULTS_DIR}/training.log"
)

# Setup device
device = setup_device("cuda")
set_seed(42)


def prepare_data():
    """Prepare datasets for training"""
    logger.info("=" * 80)
    logger.info("STEP 1: DATA PREPARATION")
    logger.info("=" * 80)

    data_loader = DataLoader(
        dataset_name=DATASET_NAME,
        split=DATASET_SPLIT,
        max_samples=MAX_SAMPLES,
        eval_size=0.1,
    )

    # Load and preprocess for each method
    logger.info("Loading raw dataset...")
    data_loader.load_dataset()

    # Save dataset locally to avoid repeated downloads
    dataset_save_path = "data/orca_dpo_pairs"
    ensure_dir(dataset_save_path)
    data_loader.raw_dataset.save_to_disk(dataset_save_path)
    logger.info(f"Dataset saved to {dataset_save_path}")

    return data_loader


# def train_reward_model(data_loader: DataLoader):
#     """Train reward model for PPO and GRPO"""
#     logger.info("=" * 80)
#     logger.info("STEP 2: REWARD MODEL TRAINING")
#     logger.info("=" * 80)

#     # Preprocess for reward model
#     data_loader.preprocess_for_reward_model()
#     train_dataset, eval_dataset = data_loader.train_test_split(test_size=0.1)

#     # Train reward model
#     reward_trainer = RewardModelTrainer(
#         model_id=BASE_MODEL_ID,
#         output_dir=CHECKPOINTS_DIR + "/reward_model",
#         load_in_8bit=True,
#     )

#     reward_trainer.load_model()

#     config = PPORewardModelConfig()
#     metrics = reward_trainer.train(
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         num_train_epochs=config.num_train_epochs,
#         per_device_train_batch_size=config.per_device_train_batch_size,
#         per_device_eval_batch_size=config.per_device_eval_batch_size,
#         learning_rate=config.learning_rate,
#         warmup_steps=config.warmup_steps,
#         weight_decay=config.weight_decay,
#         # output_dir=config.output_dir,
#     )

#     reward_trainer.save_model(config.output_dir)
#     reward_trainer.save_metrics(f"{RESULTS_DIR}/reward_model_metrics.json")

#     logger.info(f"Reward model trained and saved to {config.output_dir}")
#     return config.output_dir


# def train_dpo(data_loader: DataLoader):
#     """Train DPO alignment"""
#     logger.info("=" * 80)
#     logger.info("STEP 3: DPO TRAINING")
#     logger.info("=" * 80)

#     # Preprocess for DPO
#     data_loader.preprocess_for_dpo()
#     train_dataset, eval_dataset = data_loader.train_test_split(test_size=0.1)

#     # Train DPO
#     dpo_trainer = DPOAlignmentTrainer(
#         model_id=BASE_MODEL_ID,
#         ref_model_id=REFERENCE_MODEL_ID,
#         output_dir=CHECKPOINTS_DIR + "/dpo",
#         load_in_8bit=True,
#         use_lora=True,
#     )

#     dpo_trainer.load_models()

#     config = DPOConfig()
#     metrics = dpo_trainer.train(
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         num_train_epochs=config.num_train_epochs,
#         per_device_train_batch_size=config.per_device_train_batch_size,
#         per_device_eval_batch_size=config.per_device_eval_batch_size,
#         learning_rate=config.learning_rate,
#         beta=config.beta,
#         max_length=config.max_length,
#         max_prompt_length=config.max_prompt_length,
#         warmup_steps=config.warmup_steps,
#         weight_decay=config.weight_decay,
#         save_steps=config.save_steps,
#         eval_steps=config.eval_steps,
#         logging_steps=config.logging_steps,
#     )

#     dpo_trainer.save_model()
#     dpo_trainer.save_metrics(f"{RESULTS_DIR}/dpo_training_metrics.json")

#     logger.info("DPO training complete")
#     return dpo_trainer


# def train_ppo(data_loader: DataLoader, reward_model_path: str):
#     """Train PPO alignment"""
#     logger.info("=" * 80)
#     logger.info("STEP 4: PPO TRAINING")
#     logger.info("=" * 80)

#     # Preprocess for PPO
#     data_loader.preprocess_for_ppo()
#     train_dataset, eval_dataset = data_loader.train_test_split(test_size=0.1)

#     # Train PPO
#     ppo_trainer = PPOAlignmentTrainer(
#         model_id=BASE_MODEL_ID,
#         ref_model_id=REFERENCE_MODEL_ID,
#         reward_model_path=reward_model_path,
#         output_dir=CHECKPOINTS_DIR + "/ppo",
#         load_in_8bit=True,
#         use_lora=True,
#         reward_type="sparse",
#     )

#     ppo_trainer.load_models()

#     config = PPOConfig()
#     metrics = ppo_trainer.train(
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         num_ppo_epochs=config.num_ppo_epochs,
#         per_device_train_batch_size=config.per_device_train_batch_size,
#         per_device_eval_batch_size=config.per_device_eval_batch_size,
#         learning_rate=config.learning_rate,
#         value_learning_rate=config.value_learning_rate,
#         kl_coeff=config.kl_coeff,
#         gamma=config.gamma,
#         gae_lambda=config.gae_lambda,
#         clip_ratio=config.clip_ratio,
#         max_length=config.max_length,
#         max_prompt_length=config.max_prompt_length,
#         warmup_steps=config.warmup_steps,
#         save_steps=config.save_steps,
#         eval_steps=config.eval_steps,
#         logging_steps=config.logging_steps,
#         num_train_epochs=3,
#     )

#     ppo_trainer.save_model()
#     ppo_trainer.save_metrics(f"{RESULTS_DIR}/ppo_training_metrics.json")

#     logger.info("PPO training complete")
#     return ppo_trainer


# def train_grpo(data_loader: DataLoader, reward_model_path: str):
#     """Train GRPO alignment"""
#     logger.info("=" * 80)
#     logger.info("STEP 5: GRPO TRAINING")
#     logger.info("=" * 80)

#     # Preprocess for GRPO
#     data_loader.preprocess_for_grpo()
#     train_dataset, eval_dataset = data_loader.train_test_split(test_size=0.1)

#     # Train GRPO
#     grpo_trainer = GRPOAlignmentTrainer(
#         model_id=BASE_MODEL_ID,
#         ref_model_id=REFERENCE_MODEL_ID,
#         reward_model_path=reward_model_path,
#         output_dir=CHECKPOINTS_DIR + "/grpo",
#         load_in_8bit=True,
#         use_lora=True,
#         group_size=4,
#     )

#     grpo_trainer.load_models()

#     config = GRPOConfig()
#     metrics = grpo_trainer.train(
#         train_dataset=train_dataset,
#         num_train_epochs=config.num_train_epochs,
#         per_device_train_batch_size=config.per_device_train_batch_size,
#         learning_rate=config.learning_rate,
#         kl_coeff=config.kl_coeff,
#         max_length=config.max_length,
#         max_prompt_length=config.max_prompt_length,
#         warmup_steps=config.warmup_steps,
#         save_steps=config.save_steps,
#         logging_steps=config.logging_steps,
#     )

#     grpo_trainer.save_model()
#     grpo_trainer.save_metrics(f"{RESULTS_DIR}/grpo_training_metrics.json")

#     logger.info("GRPO training complete")
#     return grpo_trainer

def train_reward_model(data_loader: DataLoader):
    """Load pre-trained reward model for PPO and GRPO"""
    logger.info("=" * 80)
    logger.info("STEP 2: REWARD MODEL SETUP")
    logger.info("=" * 80)

    config = PPORewardModelConfig()

    if config.use_pretrained:
        logger.info(f"Using pre-trained reward model: {config.model_id}")
        logger.info("No training required - using OpenAssistant model directly")
        return config.model_id

    # Check if model already exists
    if Path(config.output_dir).exists() and (Path(config.output_dir) / "adapter_model.safetensors").exists():
        logger.info(f"Found existing reward model at {config.output_dir}, skipping training...")
        return config.output_dir

    # Preprocess for reward model
    data_loader.preprocess_for_reward_model()
    train_dataset, eval_dataset = data_loader.train_test_split(test_size=0.1)

    # Train reward model
    reward_trainer = RewardModelTrainer(
        model_id=config.model_id,
        output_dir=config.output_dir,
        load_in_8bit=True,
    )

    reward_trainer.load_model()

    metrics = reward_trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
    )

    reward_trainer.save_model(config.output_dir)
    reward_trainer.save_metrics(f"{RESULTS_DIR}/reward_model_metrics.json")

    logger.info(f"Reward model trained and saved to {config.output_dir}")
    return config.output_dir


def train_dpo(data_loader: DataLoader):
    """Train DPO alignment"""
    logger.info("=" * 80)
    logger.info("STEP 3: DPO TRAINING")
    logger.info("=" * 80)

    config = DPOConfig()
    checkpoint_dir = CHECKPOINTS_DIR + "/dpo"
    
    # Check if model already exists
    if Path(checkpoint_dir).exists() and (Path(checkpoint_dir) / "adapter_config.json").exists():
        logger.info(f"Found existing DPO model at {checkpoint_dir}, loading instead of training...")
        dpo_trainer = DPOAlignmentTrainer(
            model_id=BASE_MODEL_ID,
            ref_model_id=REFERENCE_MODEL_ID,
            output_dir=checkpoint_dir,
            load_in_8bit=True,
            use_lora=True,
        )
        dpo_trainer.load_models()
        logger.info("DPO model loaded from checkpoint")
        return dpo_trainer

    # Preprocess for DPO
    data_loader.preprocess_for_dpo()
    train_dataset, eval_dataset = data_loader.train_test_split(test_size=0.1)

    # Train DPO
    dpo_trainer = DPOAlignmentTrainer(
        model_id=BASE_MODEL_ID,
        ref_model_id=REFERENCE_MODEL_ID,
        output_dir=checkpoint_dir,
        load_in_8bit=True,
        use_lora=True,
    )

    dpo_trainer.load_models()

    metrics = dpo_trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=config.learning_rate,
        beta=config.beta,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
    )

    dpo_trainer.save_model()
    dpo_trainer.save_metrics(f"{RESULTS_DIR}/dpo_training_metrics.json")

    logger.info("DPO training complete")
    return dpo_trainer


def train_ppo(data_loader: DataLoader, reward_model_path: str):
    """Train PPO alignment"""
    logger.info("=" * 80)
    logger.info("STEP 4: PPO TRAINING")
    logger.info("=" * 80)

    checkpoint_dir = CHECKPOINTS_DIR + "/ppo"
    
    # Check if model already exists
    if Path(checkpoint_dir).exists() and (Path(checkpoint_dir) / "adapter_config.json").exists():
        logger.info(f"Found existing PPO model at {checkpoint_dir}, loading instead of training...")
        ppo_trainer = PPOAlignmentTrainer(
            model_id=BASE_MODEL_ID,
            ref_model_id=REFERENCE_MODEL_ID,
            reward_model_path=reward_model_path,
            output_dir=checkpoint_dir,
            load_in_8bit=True,
            use_lora=True,
            reward_type="sparse",
        )
        ppo_trainer.load_models()
        logger.info("PPO model loaded from checkpoint")
        return ppo_trainer

    # Preprocess for PPO
    data_loader.preprocess_for_ppo()
    train_dataset, eval_dataset = data_loader.train_test_split(test_size=0.1)

    # Train PPO
    ppo_trainer = PPOAlignmentTrainer(
        model_id=BASE_MODEL_ID,
        ref_model_id=REFERENCE_MODEL_ID,
        reward_model_path=reward_model_path,
        output_dir=checkpoint_dir,
        load_in_8bit=False,  # Disabled for SmolLM-135M (faster loading, less memory)
        use_lora=True,
        reward_type="sparse",
    )

    ppo_trainer.load_models()

    config = PPOConfig()
    metrics = ppo_trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_ppo_epochs=config.num_ppo_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=config.learning_rate,
        value_learning_rate=config.value_learning_rate,
        kl_coeff=config.kl_coeff,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_ratio=config.clip_ratio,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        warmup_steps=config.warmup_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
        num_train_epochs=3,
    )

    ppo_trainer.save_model()
    ppo_trainer.save_metrics(f"{RESULTS_DIR}/ppo_training_metrics.json")

    logger.info("PPO training complete")
    return ppo_trainer


def train_grpo(data_loader: DataLoader, reward_model_path: str):
        """Train GRPO alignment"""
        logger.info("=" * 80)
        logger.info("STEP 5: GRPO TRAINING")
        logger.info("=" * 80)

        checkpoint_dir = CHECKPOINTS_DIR + "/grpo"
        
        # Check if model already exists
        if Path(checkpoint_dir).exists() and (Path(checkpoint_dir) / "adapter_config.json").exists():
            logger.info(f"Found existing GRPO model at {checkpoint_dir}, loading instead of training...")
            grpo_trainer = GRPOAlignmentTrainer(
                model_id=BASE_MODEL_ID,
                ref_model_id=REFERENCE_MODEL_ID,
                reward_model_path=reward_model_path,
                output_dir=checkpoint_dir,
                load_in_8bit=True,
                use_lora=True,
                group_size=4,
            )
            grpo_trainer.load_models()
            logger.info("GRPO model loaded from checkpoint")
            return grpo_trainer

        # Preprocess for GRPO
        data_loader.preprocess_for_grpo()
        train_dataset, eval_dataset = data_loader.train_test_split(test_size=0.1)

        # Train GRPO
        config = GRPOConfig()
        grpo_trainer = GRPOAlignmentTrainer(
            model_id=BASE_MODEL_ID,
            ref_model_id=REFERENCE_MODEL_ID,
            reward_model_path=reward_model_path,
            output_dir=checkpoint_dir,
            load_in_8bit=False,  # Disabled for SmolLM-135M (faster loading, less memory)
            use_lora=True,
            group_size=config.group_size,  # Use config value (reduced to 2 for speed)
        )

        grpo_trainer.load_models()

        metrics = grpo_trainer.train(
            train_dataset=train_dataset,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            learning_rate=config.learning_rate,
            kl_coeff=config.kl_coeff,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            warmup_steps=config.warmup_steps,
            save_steps=config.save_steps,
            logging_steps=config.logging_steps,
        )

        grpo_trainer.save_model()
        grpo_trainer.save_metrics(f"{RESULTS_DIR}/grpo_training_metrics.json")

        logger.info("GRPO training complete")
        return grpo_trainer


def evaluate_models(data_loader: DataLoader, dpo_trainer, ppo_trainer, grpo_trainer):
    """Evaluate all trained models"""
    logger.info("=" * 80)
    logger.info("STEP 6: EVALUATION")
    logger.info("=" * 80)

    # Get evaluation prompts
    eval_prompts = data_loader.get_prompts_for_evaluation(TEST_PROMPTS_NUM)

    evaluator = EvaluationMetrics(
        ref_model_id=REFERENCE_MODEL_ID,
        device=device.type,
    )

    # Generate samples from all methods
    all_generations = {}

    for method_name, trainer in [("DPO", dpo_trainer), ("PPO", ppo_trainer), ("GRPO", grpo_trainer)]:
        logger.info(f"Evaluating {method_name}...")

        generations = {"prompts": eval_prompts, "responses": []}
        for prompt in eval_prompts[:10]:  # Evaluate on subset
            try:
                response = trainer.generate(prompt)
                generations["responses"].append(response)
            except Exception as e:
                logger.warning(f"Error generating response for {method_name}: {e}")
                generations["responses"].append("")

        all_generations[method_name] = generations

        # Save individual generations
        ensure_dir(SAMPLES_DIR)
        with open(f"{SAMPLES_DIR}/{method_name.lower()}_samples.json", 'w') as f:
            json.dump(generations, f, indent=2, ensure_ascii=False)

    # Compute evaluation metrics
    evaluation_results = {}

    # Verbosity analysis
    for method_name, generations in all_generations.items():
        if generations["responses"]:
            verbosity = evaluator.compute_verbosity_metrics(generations["responses"])
            evaluation_results[f"{method_name}_verbosity"] = verbosity
            logger.info(f"{method_name} verbosity metrics: {verbosity}")

    evaluator.save_metrics(evaluation_results, f"{RESULTS_DIR}/evaluation_metrics.json")

    # Save visualizations
    visualizer = ResultsVisualizer(output_dir=VISUALIZATIONS_DIR)
    visualizer.save_sample_generations(all_generations)

    logger.info("Evaluation complete")
    return evaluation_results


def main():
    """Main training pipeline"""
    try:
        logger.info("TASK 2: LLM ALIGNMENT")
        logger.info(f"Device: {device}")
        logger.info(f"Base Model: {BASE_MODEL_ID}")
        logger.info(f"Dataset: {DATASET_NAME}")
        logger.info("=" * 80)

        # Create output directories
        ensure_dir(CHECKPOINTS_DIR)
        ensure_dir(VISUALIZATIONS_DIR)
        ensure_dir(SAMPLES_DIR)

        # Step 1: Prepare data
        data_loader = prepare_data()

        # Step 2: Train reward model (needed for PPO and GRPO)
        reward_model_path = train_reward_model(data_loader)

        # Step 3-5: Train alignment methods
        dpo_trainer = train_dpo(data_loader)
        grpo_trainer = train_grpo(data_loader, reward_model_path)
        ppo_trainer = train_ppo(data_loader, reward_model_path)

        # Step 6: Evaluate
        evaluation_results = evaluate_models(
            data_loader, dpo_trainer, ppo_trainer, grpo_trainer
        )
        # evaluation_results = evaluate_models(
        #     data_loader, dpo_trainer, grpo_trainer
        # )
        logger.info("=" * 80)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Checkpoints saved to: {CHECKPOINTS_DIR}")
        logger.info(f"Results saved to: {RESULTS_DIR}")
        logger.info(f"Visualizations saved to: {VISUALIZATIONS_DIR}")
        logger.info(f"Samples saved to: {SAMPLES_DIR}")

    except Exception as e:
        logger.error(f"Error in training pipeline: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
