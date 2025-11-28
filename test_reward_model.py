"""
Test reward model to debug why all rewards are identical
"""
import torch
from src.modules.reward_model import RewardModelTrainer

print("="*80)
print("REWARD MODEL DEBUGGING TEST")
print("="*80)

# Initialize reward model
rm = RewardModelTrainer(
    model_id='HuggingFaceTB/SmolLM2-135M-SFT-only',
    output_dir='checkpoints/reward_model'
)

print("\nLoading reward model from checkpoints/reward_model...")
rm.load_trained_model('checkpoints/reward_model')

print("\nTesting reward model with different prompts and responses...")
print("-"*80)

# Test 1: Same prompt, different responses
test_cases = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of France?", "London"),
    ("What is the capital of France?", "Berlin"),
    ("What is 2+2?", "4"),
    ("What is 2+2?", "5"),
    ("What is 2+2?", "The answer is four."),
    ("Explain photosynthesis.", "Photosynthesis is the process by which plants convert light energy into chemical energy."),
    ("Explain photosynthesis.", "I don't know."),
    ("Explain photosynthesis.", ""),
]

rewards = []
for prompt, response in test_cases:
    reward = rm.get_reward(prompt, response)
    rewards.append(reward)
    print(f"Prompt: '{prompt[:50]}'")
    print(f"Response: '{response[:50]}'")
    print(f"Reward: {reward:.6f}")
    print()

print("-"*80)
print(f"All rewards: {rewards}")
print(f"Mean reward: {sum(rewards)/len(rewards):.6f}")
print(f"Std reward: {torch.tensor(rewards).std().item():.6f}")
print(f"Min reward: {min(rewards):.6f}")
print(f"Max reward: {max(rewards):.6f}")
print()

if torch.tensor(rewards).std().item() < 0.01:
    print("⚠️  WARNING: All rewards are nearly identical! This suggests a bug.")
    print("   The reward model should differentiate between good and bad responses.")
else:
    print("✓ Reward model is producing varied rewards.")
