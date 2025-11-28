"""
Compare Fine-tuned vs Pretrained Reward Models
Tests how reward models perform on ORCA-style preferences
"""
import json
import sys
from pathlib import Path
import numpy as np
import logging

sys.path.insert(0, str(Path(__file__).parent))

from src.modules.reward_model import RewardModelTrainer
from config import PPORewardModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_reward_model(rm: RewardModelTrainer, test_cases: list, model_name: str):
    """Test reward model on various cases"""
    results = []

    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {model_name}")
    logger.info(f"{'='*60}\n")

    for case in test_cases:
        prompt = case["prompt"]
        chosen = case["chosen"]
        rejected = case["rejected"]

        reward_chosen = rm.get_reward(prompt, chosen)
        reward_rejected = rm.get_reward(prompt, rejected)

        # Check if model prefers chosen over rejected
        correct = reward_chosen > reward_rejected
        margin = reward_chosen - reward_rejected

        result = {
            "prompt": prompt[:50] + "...",
            "reward_chosen": float(reward_chosen),
            "reward_rejected": float(reward_rejected),
            "margin": float(margin),
            "correct": bool(correct),
        }
        results.append(result)

        status = "✓" if correct else "✗"
        logger.info(f"{status} Chosen: {reward_chosen:+.4f} | Rejected: {reward_rejected:+.4f} | Margin: {margin:+.4f}")

    # Calculate accuracy
    accuracy = sum(r["correct"] for r in results) / len(results)
    avg_margin = np.mean([r["margin"] for r in results])

    logger.info(f"\n{'-'*60}")
    logger.info(f"Accuracy: {accuracy:.2%} ({sum(r['correct'] for r in results)}/{len(results)})")
    logger.info(f"Average Margin: {avg_margin:+.4f}")
    logger.info(f"{'-'*60}\n")

    return {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "average_margin": float(avg_margin),
        "results": results,
    }


def main():
    """Compare fine-tuned vs pretrained reward models"""

    # Create manual test cases
    test_cases = [
        {
            "prompt": "What is 2+2?",
            "chosen": "4",
            "rejected": "5"
        },
        {
            "prompt": "What is the capital of France?",
            "chosen": "The capital of france is Paris",
            "rejected": "London"
        },
        {
            "prompt": "Complete the sentence: The sky is",
            "chosen": "The color of sky is blue",
            "rejected": "The color of sk is purple"
        },
        {
            "prompt": "Is Python a programming language?",
            "chosen": "Yes, Python is a high-level programming language.",
            "rejected": "No, Python is a type of snake."
        },
        {
            "prompt": "What is 10 * 5?",
            "chosen": "50",
            "rejected": "15"
        },
        {
            "prompt": "Generate an approximately fifteen-word sentence describing a restaurant.",
            "chosen": "Midsummer House is a moderately priced Chinese restaurant with a 3/5 rating.",
            "rejected": "Sure! Here's a sentence that describes a restaurant: Midsummer House is a moderately priced Chinese restaurant with a 3/5 customer rating, located near All Bar One."
        },
        {
            "prompt": "What happens if you add 3 + 7?",
            "chosen": "10",
            "rejected": "I think it might be around 11 or 12, but I'm not completely sure about this calculation."
        },
        {
            "prompt": "Explain photosynthesis briefly.",
            "chosen": "Photosynthesis is the process where plants convert sunlight, water, and CO2 into glucose and oxygen.",
            "rejected": "Well, you know, photosynthesis is like this really complex thing that plants do, and I'm happy to help explain it! So basically, what happens is that plants use sunlight, and they take in water and carbon dioxide, and through a series of really intricate chemical reactions that involve chlorophyll and other pigments, they eventually produce glucose which is a sugar that gives them energy, and they also release oxygen as a byproduct which is great for us!"
        },
    ]

    comparison_results = {}

    # Test 1: Fine-tuned model (trained on ORCA)
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 1: Fine-tuned Reward Model (Trained on ORCA)")
    logger.info("="*60)

    try:
        rm_finetuned = RewardModelTrainer(
            model_id="OpenAssistant/reward-model-deberta-v3-large",
            output_dir="checkpoints/reward_model",
            load_in_8bit=True,
        )

        # Check if fine-tuned model exists
        finetuned_path = Path("checkpoints/reward_model")
        if finetuned_path.exists() and (finetuned_path / "config.json").exists():
            logger.info("Loading fine-tuned model from checkpoint...")
            rm_finetuned.load_trained_model("checkpoints/reward_model")
        else:
            logger.warning("Fine-tuned model not found. Run main.py to train it first.")
            logger.info("Skipping fine-tuned model test...")
            comparison_results["finetuned"] = None
    except Exception as e:
        logger.error(f"Error loading fine-tuned model: {e}")
        comparison_results["finetuned"] = None

    if comparison_results.get("finetuned") is not None or "finetuned" not in comparison_results:
        if "finetuned" not in comparison_results:
            try:
                finetuned_results = test_reward_model(
                    rm_finetuned,
                    test_cases,
                    "Fine-tuned on ORCA"
                )
                comparison_results["finetuned"] = finetuned_results
            except Exception as e:
                logger.error(f"Error testing fine-tuned model: {e}")
                comparison_results["finetuned"] = None

    # Test 2: Pretrained model (no fine-tuning)
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 2: Pretrained Reward Model (Zero-shot)")
    logger.info("="*60)

    try:
        rm_pretrained = RewardModelTrainer(
            model_id="OpenAssistant/reward-model-deberta-v3-large",
            output_dir="checkpoints/reward_model_pretrained",
            load_in_8bit=True,
        )

        logger.info("Loading pretrained model (no fine-tuning)...")
        # Load directly from HuggingFace
        rm_pretrained.load_model()

        pretrained_results = test_reward_model(
            rm_pretrained,
            test_cases,
            "Pretrained (Zero-shot)"
        )
        comparison_results["pretrained"] = pretrained_results

    except Exception as e:
        logger.error(f"Error testing pretrained model: {e}")
        import traceback
        traceback.print_exc()
        comparison_results["pretrained"] = None

    # Compare results
    logger.info("\n" + "="*60)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*60 + "\n")

    if comparison_results.get("finetuned") and comparison_results.get("pretrained"):
        ft = comparison_results["finetuned"]
        pt = comparison_results["pretrained"]

        logger.info(f"Fine-tuned Accuracy:  {ft['accuracy']:.2%}")
        logger.info(f"Pretrained Accuracy:  {pt['accuracy']:.2%}")
        logger.info(f"Improvement:          {(ft['accuracy'] - pt['accuracy'])*100:+.1f}%\n")

        logger.info(f"Fine-tuned Avg Margin: {ft['average_margin']:+.4f}")
        logger.info(f"Pretrained Avg Margin: {pt['average_margin']:+.4f}")
        logger.info(f"Difference:            {(ft['average_margin'] - pt['average_margin']):+.4f}\n")

        if ft['accuracy'] > pt['accuracy']:
            logger.info("✓ Fine-tuning improved accuracy!")
        elif ft['accuracy'] < pt['accuracy']:
            logger.info("✗ Pretrained model was more accurate")
        else:
            logger.info("= Both models have same accuracy")

    # Save results
    output_file = "test_results/reward_model_comparison.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
