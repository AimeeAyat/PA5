"""Quick test to verify reward model fix"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing reward model with probability-based rewards...")
print("="*60)

try:
    from src.modules.reward_model import RewardModelTrainer

    rm = RewardModelTrainer(
        model_id='HuggingFaceTB/SmolLM2-135M-SFT-only',
        output_dir='checkpoints/reward_model'
    )
    rm.load_trained_model('checkpoints/reward_model')

    test_cases = [
        ("What is 2+2?", "4", "correct"),
        ("What is 2+2?", "5", "wrong"),
        ("What is 2+2?", "The answer is 10", "very wrong"),
        ("What is the capital of France?", "Paris", "correct"),
        ("What is the capital of France?", "London", "wrong"),
        ("What is the capital of France?", "I don't know", "uncertain"),
    ]

    print("\nTest Results:")
    print("-" * 60)

    correct_rewards = []
    wrong_rewards = []

    for prompt, response, label in test_cases:
        reward = rm.get_reward(prompt, response)

        if label == "correct":
            correct_rewards.append(reward)
            marker = "[CORRECT]"
        elif label == "wrong" or label == "very wrong":
            wrong_rewards.append(reward)
            marker = "[WRONG]  "
        else:
            marker = "[OTHER]  "

        print(f"{marker} {prompt[:30]:30} -> {response:20}: {reward:+.6f}")

    print("\n" + "=" * 60)
    print("ANALYSIS:")
    print("=" * 60)

    import numpy as np

    if correct_rewards and wrong_rewards:
        avg_correct = np.mean(correct_rewards)
        avg_wrong = np.mean(wrong_rewards)

        print(f"Average reward for CORRECT answers: {avg_correct:+.6f}")
        print(f"Average reward for WRONG answers:   {avg_wrong:+.6f}")
        print(f"Difference: {avg_correct - avg_wrong:+.6f}")

        if avg_correct > avg_wrong:
            print("\n✓ PASS: Correct answers get higher rewards on average!")
        else:
            print("\n✗ FAIL: Wrong answers still getting higher rewards")
            print("   -> Reward model may need retraining")

    all_rewards = correct_rewards + wrong_rewards
    reward_std = np.std(all_rewards)

    print(f"\nReward standard deviation: {reward_std:.6f}")

    if reward_std > 0.01:
        print("✓ PASS: Rewards are diverse (std > 0.01)")
    else:
        print("✗ FAIL: Rewards are still too similar")

    print("\n" + "=" * 60)
    print("NOTE: Rewards are now in range [-1, 1]")
    print("  +1.0 = strongly prefer 'chosen'")
    print("   0.0 = uncertain")
    print("  -1.0 = strongly prefer 'rejected'")
    print("=" * 60)

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
