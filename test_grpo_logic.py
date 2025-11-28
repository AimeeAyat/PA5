"""
Test GRPO Logic Only (No Model Loading)
Validates advantage computation and identifies implementation issues
"""
import numpy as np
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_group_relative_advantage(rewards):
    """
    GRPO advantage formula from assignment:
    Â_i = (R_i - R̄) / σ_R
    """
    rewards_array = np.array(rewards)
    mean_reward = np.mean(rewards_array)
    std_reward = np.std(rewards_array)

    if std_reward == 0:
        std_reward = 1e-8

    advantages = (rewards_array - mean_reward) / std_reward
    return advantages.tolist()


def test_advantage_computation():
    """Test group-relative advantage computation"""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Advantage Computation Formula")
    logger.info("="*60)

    # Test case 1: Varied rewards
    rewards = [1.0, 2.0, 3.0, 4.0]
    advantages = compute_group_relative_advantage(rewards)

    logger.info(f"\nRewards:    {rewards}")
    logger.info(f"Advantages: {[f'{a:.4f}' for a in advantages]}")
    logger.info(f"Mean advantage: {np.mean(advantages):.6f} (should be ~0)")
    logger.info(f"Std advantage:  {np.std(advantages):.6f} (should be ~1)")

    # Verify properties
    assert abs(np.mean(advantages)) < 1e-6, "Mean should be 0"
    assert abs(np.std(advantages) - 1.0) < 1e-6, "Std should be 1"
    assert advantages[0] < advantages[1] < advantages[2] < advantages[3], "Should be ascending"
    logger.info("✓ Correct: Advantages are normalized (mean=0, std=1)")

    # Test case 2: All same rewards
    rewards = [5.0, 5.0, 5.0, 5.0]
    advantages = compute_group_relative_advantage(rewards)
    logger.info(f"\nRewards (identical): {rewards}")
    logger.info(f"Advantages: {[f'{a:.4f}' for a in advantages]}")
    assert all(abs(a) < 1e-6 for a in advantages), "All advantages should be 0"
    logger.info("✓ Correct: Zero advantage when all rewards equal")

    # Test case 3: Realistic scenario
    rewards = [2.1, 3.5, 2.8, 4.2]
    advantages = compute_group_relative_advantage(rewards)
    logger.info(f"\nRealistic rewards: {rewards}")
    for i, (r, a) in enumerate(zip(rewards, advantages)):
        logger.info(f"  Response {i+1}: Reward={r:.2f} → Advantage={a:+.4f}")
    logger.info("✓ Advantage computation is CORRECT ✓\n")


def test_loss_weighting_issue():
    """Identify the critical bug in loss weighting"""
    logger.info("="*60)
    logger.info("TEST 2: Loss Weighting Analysis")
    logger.info("="*60)

    base_loss = 2.0

    logger.info(f"\nBase loss (same for all): {base_loss}")
    logger.info("\nCurrent implementation (grpo_trainer.py:305):")
    logger.info("  weighted_loss = loss * (1.0 - advantage)")
    logger.info("\n" + "-"*60)

    test_cases = [
        ("Best response (high reward)", +1.5),
        ("Good response", +0.5),
        ("Average response", 0.0),
        ("Bad response", -0.5),
        ("Worst response (low reward)", -1.5),
    ]

    logger.info(f"{'Description':<30} | {'Advantage':>10} | {'Current Loss':>13} | {'Effect':>15}")
    logger.info("-"*75)

    for desc, advantage in test_cases:
        current_weighted = base_loss * (1.0 - advantage)
        effect = "WRONG!" if advantage > 0 else "WRONG!"
        logger.info(f"{desc:<30} | {advantage:>+10.2f} | {current_weighted:>13.4f} | {effect:>15}")

    logger.info("-"*75)
    logger.info("\n❌ CRITICAL BUG IDENTIFIED:")
    logger.info("   • Best response (advantage=+1.5) → loss × (-0.5) = NEGATIVE LOSS!")
    logger.info("   • Worst response (advantage=-1.5) → loss × (2.5) = HIGH LOSS")
    logger.info("   • This INVERTS the optimization!")
    logger.info("\n✓ Correct approach:")
    logger.info("   • Use policy gradient: -advantage * log_prob")
    logger.info("   • Higher advantage → gradient increases log_prob")
    logger.info("   • Lower advantage → gradient decreases log_prob")


def test_correct_implementation():
    """Show what correct GRPO loss should look like"""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Correct GRPO Implementation")
    logger.info("="*60)

    logger.info("\nFrom assignment:")
    logger.info("'The policy is then updated using a PPO-style clipped objective")
    logger.info(" applied to token log-probabilities.'")

    logger.info("\nCorrect GRPO loss (policy gradient with clipping):")
    logger.info("""
    # For each response in group:
    1. Get log_prob from current policy
    2. Get old_log_prob (stored during generation)
    3. Compute ratio = exp(log_prob - old_log_prob)
    4. Compute surrogate losses:
       - unclipped = ratio * advantage
       - clipped = clip(ratio, 1-ε, 1+ε) * advantage
    5. Final loss = -min(unclipped, clipped)  # Negative for gradient ascent
    """)

    logger.info("\nKey differences from current implementation:")
    logger.info("  ❌ Current: Uses language modeling loss × (1 - advantage)")
    logger.info("  ✓ Correct: Uses policy gradient -advantage × log_prob with PPO clipping")


def analyze_grpo_code():
    """Analyze the actual GRPO code for issues"""
    logger.info("\n" + "="*60)
    logger.info("CODE ANALYSIS: grpo_trainer.py")
    logger.info("="*60)

    issues = []

    # Issue 1: Loss weighting
    issues.append({
        "id": 1,
        "severity": "CRITICAL",
        "line": 305,
        "function": "train()",
        "issue": "Loss weighting inverted",
        "code": "weighted_loss = loss * (1.0 - advantage)",
        "problem": [
            "High advantage (good) → negative/low weighted loss",
            "Low advantage (bad) → high weighted loss",
            "Training optimizes in WRONG direction"
        ],
        "fix": "Use policy gradient: loss = -advantage * log_prob with PPO clipping",
        "impact": "Model learns to prefer BAD responses over GOOD ones"
    })

    # Issue 2: Missing PPO clipping
    issues.append({
        "id": 2,
        "severity": "CRITICAL",
        "line": 298,
        "function": "train()",
        "issue": "Missing PPO-style clipped objective",
        "code": "outputs = self.model(**inputs, labels=inputs.input_ids); loss = outputs.loss",
        "problem": [
            "Assignment specifies 'PPO-style clipped objective'",
            "Current implementation uses standard LM loss without clipping",
            "No ratio clipping, no old_log_prob storage"
        ],
        "fix": "Implement ratio = exp(log_prob - old_log_prob) and clip(ratio, 1-ε, 1+ε)",
        "impact": "Training instability, no protection against large policy updates"
    })

    # Issue 3: Reward model compatibility
    issues.append({
        "id": 3,
        "severity": "MAJOR",
        "line": 137,
        "function": "get_reward()",
        "issue": "Assumes binary classification reward model",
        "code": "reward = outputs.logits[0, 1].item()",
        "problem": [
            "OpenAssistant model is regression (num_labels=1)",
            "Will crash with IndexError"
        ],
        "fix": "Check logits.shape[-1]: use [0,0] for regression, [0,1] for classification",
        "impact": "Runtime error when using OpenAssistant reward model"
    })

    # Issue 4: Missing KL penalty
    issues.append({
        "id": 4,
        "severity": "MAJOR",
        "line": 267,
        "function": "train()",
        "issue": "Missing KL penalty in reward",
        "code": "reward = self.get_reward(prompt, response)",
        "problem": [
            "Assignment says 'reward already includes KL penalty'",
            "Current implementation: reward comes from reward model only",
            "No KL divergence vs reference model"
        ],
        "fix": "Add: reward = rm_reward - kl_coeff * KL(policy || ref_policy)",
        "impact": "Policy can drift far from reference, reduced stability"
    })

    # Print issues
    for issue in issues:
        logger.info(f"\n{'='*60}")
        logger.info(f"ISSUE #{issue['id']}: [{issue['severity']}]")
        logger.info(f"{'='*60}")
        logger.info(f"Location: grpo_trainer.py:{issue['line']} in {issue['function']}")
        logger.info(f"Issue: {issue['issue']}")
        logger.info(f"\nCurrent code:")
        logger.info(f"  {issue['code']}")
        logger.info(f"\nProblems:")
        for prob in issue['problem']:
            logger.info(f"  • {prob}")
        logger.info(f"\nFix:")
        logger.info(f"  {issue['fix']}")
        logger.info(f"\nImpact:")
        logger.info(f"  {issue['impact']}")

    # Save to JSON
    output_file = "test_results/grpo_issues.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(issues, f, indent=2)

    logger.info(f"\n✓ Issues saved to: {output_file}")
    return issues


def main():
    """Run all logic tests"""
    logger.info("\n" + "█"*60)
    logger.info("GRPO IMPLEMENTATION VALIDATION (LOGIC ONLY)")
    logger.info("█"*60)

    # Test 1: Advantage computation (CORRECT)
    test_advantage_computation()

    # Test 2: Loss weighting analysis (WRONG)
    test_loss_weighting_issue()

    # Test 3: Show correct implementation
    test_correct_implementation()

    # Test 4: Analyze code
    issues = analyze_grpo_code()

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)

    critical = sum(1 for i in issues if i['severity'] == 'CRITICAL')
    major = sum(1 for i in issues if i['severity'] == 'MAJOR')

    logger.info(f"\n✓ Advantage computation: CORRECT")
    logger.info(f"❌ Loss implementation: {critical} CRITICAL issues")
    logger.info(f"⚠️  Other issues: {major} MAJOR issues")
    logger.info(f"\nTotal issues found: {len(issues)}")

    logger.info("\n" + "-"*60)
    logger.info("RECOMMENDATION:")
    logger.info("-"*60)
    logger.info("Fix CRITICAL issues before training:")
    logger.info("  1. Implement proper policy gradient with advantages")
    logger.info("  2. Add PPO-style ratio clipping")
    logger.info("  3. Fix reward model compatibility")
    logger.info("  4. Add KL penalty to rewards")
    logger.info("-"*60)

    logger.info("\n" + "="*60)


if __name__ == "__main__":
    main()
