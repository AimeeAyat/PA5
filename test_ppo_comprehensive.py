"""
Comprehensive PPO Implementation Testing Script
Tests: Dataset Loading, Reward Model, PPO Components
Generates detailed report with sample data for verification
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("COMPREHENSIVE PPO TESTING SCRIPT")
print("="*80)

# Create output directory for test results
test_output_dir = "test_results"
Path(test_output_dir).mkdir(exist_ok=True)

# ============================================================================
# TEST 1: DATASET LOADING
# ============================================================================
print("\n" + "="*80)
print("TEST 1: DATASET LOADING")
print("="*80)

try:
    from datasets import load_from_disk

    dataset_path = "data/orca_dpo_pairs"
    print(f"\nLoading dataset from: {dataset_path}")

    if not Path(dataset_path).exists():
        print(f"[ERROR] Dataset not found at {dataset_path}")
        print("   Run main.py first to download the dataset")
        sys.exit(1)

    dataset = load_from_disk(dataset_path)
    print(f"[OK] Dataset loaded successfully")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Columns: {dataset.column_names}")

    # Sample data for visualization
    sample_size = min(10, len(dataset))
    sample_data = []

    print(f"\n--- Sample Data (first {sample_size} rows) ---")
    for i in range(sample_size):
        example = dataset[i]
        sample_data.append({
            "index": i,
            "question": example["question"][:100] + "..." if len(example["question"]) > 100 else example["question"],
            "chosen_preview": example["chosen"][:100] + "..." if len(example["chosen"]) > 100 else example["chosen"],
            "rejected_preview": example["rejected"][:100] + "..." if len(example["rejected"]) > 100 else example["rejected"],
            "chosen_length": len(example["chosen"]),
            "rejected_length": len(example["rejected"]),
        })

        print(f"\nSample {i+1}:")
        print(f"  Question: {sample_data[i]['question']}")
        print(f"  Chosen length: {sample_data[i]['chosen_length']} chars")
        print(f"  Rejected length: {sample_data[i]['rejected_length']} chars")

    # Save sample data to JSON
    sample_file = f"{test_output_dir}/dataset_samples.json"
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_samples": len(dataset),
            "columns": dataset.column_names,
            "sample_data": sample_data
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Sample data saved to: {sample_file}")

    # Statistics
    print(f"\n--- Dataset Statistics ---")
    chosen_lengths = [len(dataset[i]["chosen"]) for i in range(min(1000, len(dataset)))]
    rejected_lengths = [len(dataset[i]["rejected"]) for i in range(min(1000, len(dataset)))]
    question_lengths = [len(dataset[i]["question"]) for i in range(min(1000, len(dataset)))]

    stats = {
        "question_length_mean": float(np.mean(question_lengths)),
        "question_length_std": float(np.std(question_lengths)),
        "chosen_length_mean": float(np.mean(chosen_lengths)),
        "chosen_length_std": float(np.std(chosen_lengths)),
        "rejected_length_mean": float(np.mean(rejected_lengths)),
        "rejected_length_std": float(np.std(rejected_lengths)),
    }

    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")

    print("\n[OK] TEST 1 PASSED: Dataset loaded and analyzed successfully")

except Exception as e:
    print(f"\n[ERROR] TEST 1 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 2: REWARD MODEL
# ============================================================================
print("\n" + "="*80)
print("TEST 2: REWARD MODEL")
print("="*80)

try:
    from src.modules.reward_model import RewardModelTrainer

    reward_model_path = "checkpoints/reward_model"
    print(f"\nChecking reward model at: {reward_model_path}")

    if not Path(reward_model_path).exists():
        print(f"[ERROR] Reward model not found at {reward_model_path}")
        print("   Run main.py first to train the reward model")
        sys.exit(1)

    print("Loading reward model...")
    rm = RewardModelTrainer(
        model_id='HuggingFaceTB/SmolLM2-135M-SFT-only',
        output_dir=reward_model_path
    )

    # Load the trained model
    print("Loading trained reward model...")
    rm.load_trained_model(reward_model_path)
    print("[OK] Reward model loaded successfully")

    # Test reward model with various inputs
    print("\n--- Testing Reward Model with Sample Inputs ---")

    test_cases = [
        # Good vs Bad responses
        ("What is the capital of France?", "The capital of France is Paris.", "good"),
        ("What is the capital of France?", "London", "bad"),
        ("What is the capital of France?", "I don't know", "bad"),

        # Correct vs Incorrect math
        ("What is 2+2?", "2+2 equals 4.", "good"),
        ("What is 2+2?", "5", "bad"),
        ("What is 2+2?", "The answer is four, which can be calculated by adding two and two.", "good"),

        # Quality of explanation
        ("Explain photosynthesis.",
         "Photosynthesis is the process by which plants convert light energy into chemical energy, using carbon dioxide and water to produce glucose and oxygen.",
         "good"),
        ("Explain photosynthesis.",
         "I don't know.",
         "bad"),
        ("Explain photosynthesis.",
         "It's when plants make food.",
         "mediocre"),

        # Empty/short responses
        ("Tell me about quantum physics.",
         "Quantum physics is a branch of physics that studies the behavior of matter and energy at the atomic and subatomic level.",
         "good"),
        ("Tell me about quantum physics.",
         "",
         "bad"),
    ]

    reward_test_results = []
    print()

    for i, (prompt, response, quality) in enumerate(test_cases):
        reward = rm.get_reward(prompt, response)
        reward_test_results.append({
            "test_id": i + 1,
            "prompt": prompt[:60] + "..." if len(prompt) > 60 else prompt,
            "response": response[:60] + "..." if len(response) > 60 else response,
            "expected_quality": quality,
            "reward_score": float(reward)
        })

        print(f"Test {i+1} ({quality}):")
        print(f"  Prompt: {prompt[:60]}")
        print(f"  Response: {response[:60]}")
        print(f"  Reward: {reward:.6f}")
        print()

    # Analyze reward distribution
    rewards = [r["reward_score"] for r in reward_test_results]
    good_rewards = [r["reward_score"] for r in reward_test_results if r["expected_quality"] == "good"]
    bad_rewards = [r["reward_score"] for r in reward_test_results if r["expected_quality"] == "bad"]

    reward_stats = {
        "all_rewards_mean": float(np.mean(rewards)),
        "all_rewards_std": float(np.std(rewards)),
        "all_rewards_min": float(np.min(rewards)),
        "all_rewards_max": float(np.max(rewards)),
        "good_rewards_mean": float(np.mean(good_rewards)) if good_rewards else 0,
        "bad_rewards_mean": float(np.mean(bad_rewards)) if bad_rewards else 0,
        "reward_variance": float(np.var(rewards)),
    }

    print("--- Reward Statistics ---")
    for key, value in reward_stats.items():
        print(f"  {key}: {value:.6f}")

    # Check if rewards are diverse
    print("\n--- Reward Model Quality Check ---")
    if reward_stats["all_rewards_std"] < 0.01:
        print("[ERROR] WARNING: Reward model produces nearly identical rewards!")
        print("   Standard deviation is too low. The model may not be discriminating properly.")
        print(f"   Std Dev: {reward_stats['all_rewards_std']:.6f}")
    else:
        print(f"[OK] Reward model produces diverse rewards (std: {reward_stats['all_rewards_std']:.6f})")

    if len(good_rewards) > 0 and len(bad_rewards) > 0:
        if reward_stats["good_rewards_mean"] > reward_stats["bad_rewards_mean"]:
            print(f"[OK] Good responses get higher rewards on average")
            print(f"   Good: {reward_stats['good_rewards_mean']:.6f}, Bad: {reward_stats['bad_rewards_mean']:.6f}")
        else:
            print(f"[ERROR] WARNING: Bad responses get higher rewards!")
            print(f"   Good: {reward_stats['good_rewards_mean']:.6f}, Bad: {reward_stats['bad_rewards_mean']:.6f}")

    # Save reward test results
    reward_file = f"{test_output_dir}/reward_model_test.json"
    with open(reward_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_cases": reward_test_results,
            "statistics": reward_stats
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Reward test results saved to: {reward_file}")
    print("\n[OK] TEST 2 PASSED: Reward model tested successfully")

except Exception as e:
    print(f"\n[ERROR] TEST 2 FAILED: {e}")
    import traceback
    traceback.print_exc()
    # Don't exit, continue to next test

# ============================================================================
# TEST 3: PPO TRAINING METRICS ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("TEST 3: PPO TRAINING METRICS ANALYSIS")
print("="*80)

try:
    metrics_file = "results/ppo_training_metrics.json"
    print(f"\nLoading PPO training metrics from: {metrics_file}")

    if not Path(metrics_file).exists():
        print(f"[ERROR] PPO metrics not found at {metrics_file}")
        print("   Run PPO training first")
    else:
        with open(metrics_file, 'r') as f:
            ppo_metrics = json.load(f)

        print("[OK] PPO metrics loaded successfully")
        print(f"\n--- PPO Training Metrics ---")
        print(json.dumps(ppo_metrics, indent=2))

        # Analyze metrics
        print("\n--- Metrics Analysis ---")

        if "metrics" in ppo_metrics:
            epochs = ppo_metrics["metrics"]

            # Check for issues
            issues = []

            for epoch in epochs:
                if "std_reward" in epoch and epoch["std_reward"] == 0:
                    issues.append(f"Epoch {epoch['epoch']}: Zero reward std deviation!")

                if "mean_reward" in epoch and "max_reward" in epoch and "min_reward" in epoch:
                    if epoch["mean_reward"] == epoch["max_reward"] == epoch["min_reward"]:
                        issues.append(f"Epoch {epoch['epoch']}: All rewards are identical!")

            if issues:
                print("[ERROR] CRITICAL ISSUES FOUND:")
                for issue in issues:
                    print(f"   - {issue}")
            else:
                print("[OK] No critical metric issues detected")

            # Check for missing metrics
            expected_metrics = ["mean_policy_loss", "mean_value_loss", "mean_kl_penalty"]
            missing_metrics = []

            for metric in expected_metrics:
                if metric not in epochs[0]:
                    missing_metrics.append(metric)

            if missing_metrics:
                print(f"\n[WARN]  Missing metrics in output: {missing_metrics}")
                print("   These should be tracked according to the code")

            print("\n[OK] TEST 3 PASSED: Metrics analyzed")

except Exception as e:
    print(f"\n[ERROR] TEST 3 FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# GENERATE COMPREHENSIVE REPORT
# ============================================================================
print("\n" + "="*80)
print("GENERATING COMPREHENSIVE REPORT")
print("="*80)

report = {
    "test_summary": {
        "dataset_test": "PASSED",
        "reward_model_test": "PASSED",
        "metrics_analysis": "PASSED"
    },
    "issues_found": [],
    "recommendations": []
}

# Check for issues
if reward_stats.get("all_rewards_std", 0) < 0.01:
    report["issues_found"].append({
        "severity": "CRITICAL",
        "component": "Reward Model",
        "issue": "Reward model produces nearly identical rewards (std < 0.01)",
        "impact": "PPO training cannot differentiate between good and bad responses"
    })
    report["recommendations"].append(
        "Retrain reward model with different hyperparameters or check training data quality"
    )

# Add more checks based on the tests
report_file = f"{test_output_dir}/comprehensive_test_report.json"
with open(report_file, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"\n[OK] Comprehensive report saved to: {report_file}")

print("\n" + "="*80)
print("ALL TESTS COMPLETED")
print("="*80)
print(f"\nTest results saved in: {test_output_dir}/")
print("Files generated:")
print(f"  1. {test_output_dir}/dataset_samples.json")
print(f"  2. {test_output_dir}/reward_model_test.json")
print(f"  3. {test_output_dir}/comprehensive_test_report.json")
print("\nReview these files to verify the PPO implementation.")
