"""
Test GRPO Implementation
Validates core GRPO logic and components
"""
import sys
from pathlib import Path
import numpy as np
import json
import logging

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_advantage_computation():
    """Test group-relative advantage computation"""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Advantage Computation")
    logger.info("="*60)

    from src.trainers.grpo_trainer import GRPOAlignmentTrainer

    trainer = GRPOAlignmentTrainer()

    # Test case 1: Varied rewards
    rewards = [1.0, 2.0, 3.0, 4.0]
    advantages = trainer.compute_group_relative_advantage(rewards)

    logger.info(f"Rewards:    {rewards}")
    logger.info(f"Advantages: {[f'{a:.4f}' for a in advantages]}")
    logger.info(f"Mean advantage: {np.mean(advantages):.6f} (should be ~0)")
    logger.info(f"Std advantage: {np.std(advantages):.6f} (should be ~1)")

    # Verify properties
    assert abs(np.mean(advantages)) < 1e-6, "Mean should be 0"
    assert abs(np.std(advantages) - 1.0) < 1e-6, "Std should be 1"
    assert advantages[0] < advantages[1] < advantages[2] < advantages[3], "Should be ascending"
    logger.info("✓ Test case 1 passed: Varied rewards\n")

    # Test case 2: All same rewards
    rewards = [5.0, 5.0, 5.0, 5.0]
    advantages = trainer.compute_group_relative_advantage(rewards)

    logger.info(f"Rewards (all same): {rewards}")
    logger.info(f"Advantages: {[f'{a:.4f}' for a in advantages]}")
    logger.info(f"All advantages should be 0 when rewards are identical")

    assert all(abs(a) < 1e-6 for a in advantages), "All advantages should be 0"
    logger.info("✓ Test case 2 passed: Identical rewards\n")

    # Test case 3: One outlier
    rewards = [1.0, 1.0, 1.0, 10.0]
    advantages = trainer.compute_group_relative_advantage(rewards)

    logger.info(f"Rewards (one outlier): {rewards}")
    logger.info(f"Advantages: {[f'{a:.4f}' for a in advantages]}")
    logger.info(f"Last advantage: {advantages[3]:.4f} (should be large positive)")

    assert advantages[3] > 1.0, "Outlier should have large positive advantage"
    logger.info("✓ Test case 3 passed: Outlier detection\n")


def test_loss_weighting():
    """Test if loss weighting makes sense"""
    logger.info("="*60)
    logger.info("TEST 2: Loss Weighting Logic")
    logger.info("="*60)

    # Simulate GRPO loss weighting
    base_loss = 2.0

    test_cases = [
        ("High advantage (good response)", 1.5),
        ("Medium advantage", 0.5),
        ("Zero advantage (average)", 0.0),
        ("Negative advantage (bad response)", -1.5),
    ]

    logger.info("\nCurrent implementation: weighted_loss = loss * (1.0 - advantage)\n")

    for desc, advantage in test_cases:
        weighted_loss = base_loss * (1.0 - advantage)
        logger.info(f"{desc:40s} | Advantage: {advantage:+.2f} | Weighted Loss: {weighted_loss:.4f}")

    logger.info("\n⚠️  ISSUE DETECTED:")
    logger.info("High advantage (good) → HIGHER weighted loss (wrong!)")
    logger.info("Low advantage (bad) → LOWER weighted loss (wrong!)")
    logger.info("\nCorrect behavior should be:")
    logger.info("High advantage (good) → LOWER effective loss")
    logger.info("Low advantage (bad) → HIGHER effective loss")

    logger.info("\n" + "-"*60)
    logger.info("Suggested fix: weighted_loss = loss * (1.0 + advantage)")
    logger.info("Or better: Use policy gradient with -advantage * log_prob")
    logger.info("-"*60 + "\n")


def test_grpo_components():
    """Test individual GRPO components without full training"""
    logger.info("="*60)
    logger.info("TEST 3: Component Integration")
    logger.info("="*60)

    from src.trainers.grpo_trainer import GRPOAlignmentTrainer

    # Test initialization
    trainer = GRPOAlignmentTrainer(
        model_id="HuggingFaceTB/SmolLM2-135M-SFT-only",
        group_size=4,
    )

    logger.info(f"✓ Model ID: {trainer.model_id}")
    logger.info(f"✓ Group size: {trainer.group_size}")
    logger.info(f"✓ Use LoRA: {trainer.use_lora}")
    logger.info(f"✓ Load in 8-bit: {trainer.load_in_8bit}")

    # Test advantage computation with realistic reward values
    logger.info("\nSimulating rewards from 4 responses:")
    realistic_rewards = [2.3, 3.1, 2.8, 4.2]
    advantages = trainer.compute_group_relative_advantage(realistic_rewards)

    for i, (r, a) in enumerate(zip(realistic_rewards, advantages)):
        logger.info(f"  Response {i+1}: Reward={r:.2f}, Advantage={a:+.4f}")

    logger.info("\n✓ All components initialized correctly")


def test_grpo_issues():
    """Document all issues found in GRPO implementation"""
    logger.info("\n" + "="*60)
    logger.info("GRPO IMPLEMENTATION ANALYSIS")
    logger.info("="*60)

    issues = [
        {
            "id": 1,
            "severity": "CRITICAL",
            "location": "grpo_trainer.py:305",
            "issue": "Loss weighting is inverted",
            "current": "weighted_loss = loss * (1.0 - advantage)",
            "problem": "High-advantage responses get HIGHER loss (wrong direction)",
            "fix": "Use policy gradient: loss = -advantage * log_prob, or weighted_loss = loss * (1.0 + advantage)"
        },
        {
            "id": 2,
            "severity": "MAJOR",
            "location": "grpo_trainer.py:137",
            "issue": "Reward extraction assumes binary classification",
            "current": "reward = outputs.logits[0, 1].item()",
            "problem": "Fails with regression models (num_labels=1) like OpenAssistant",
            "fix": "Check logits.shape[-1] and use [0,0] for regression, [0,1] for classification"
        },
        {
            "id": 3,
            "severity": "MINOR",
            "location": "grpo_trainer.py",
            "issue": "KL coefficient parameter not used",
            "current": "kl_coeff parameter passed but never applied",
            "problem": "Missing KL divergence penalty vs reference model",
            "fix": "Add KL penalty term to loss"
        },
        {
            "id": 4,
            "severity": "INFO",
            "location": "grpo_trainer.py:298-299",
            "issue": "Using language modeling loss instead of policy gradient",
            "current": "outputs = self.model(**inputs, labels=inputs.input_ids); loss = outputs.loss",
            "problem": "Standard LM loss, not explicit policy gradient",
            "fix": "This is acceptable but less direct than computing log_probs explicitly"
        }
    ]

    for issue in issues:
        logger.info(f"\nISSUE #{issue['id']}: [{issue['severity']}] {issue['issue']}")
        logger.info(f"Location: {issue['location']}")
        logger.info(f"Current:  {issue['current']}")
        logger.info(f"Problem:  {issue['problem']}")
        logger.info(f"Fix:      {issue['fix']}")

    # Save to JSON
    output_file = "test_results/grpo_issues.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(issues, f, indent=2)

    logger.info(f"\n✓ Issues saved to: {output_file}")

    return issues


def main():
    """Run all GRPO tests"""
    logger.info("\n" + "█"*60)
    logger.info("GRPO IMPLEMENTATION VALIDATION")
    logger.info("█"*60)

    try:
        # Test 1: Advantage computation
        test_advantage_computation()

        # Test 2: Loss weighting analysis
        test_loss_weighting()

        # Test 3: Components
        test_grpo_components()

        # Test 4: Document all issues
        issues = test_grpo_issues()

        # Summary
        logger.info("\n" + "="*60)
        logger.info("SUMMARY")
        logger.info("="*60)

        critical = sum(1 for i in issues if i['severity'] == 'CRITICAL')
        major = sum(1 for i in issues if i['severity'] == 'MAJOR')
        minor = sum(1 for i in issues if i['severity'] == 'MINOR')

        logger.info(f"Critical issues: {critical}")
        logger.info(f"Major issues:    {major}")
        logger.info(f"Minor issues:    {minor}")
        logger.info(f"Total issues:    {len(issues)}")

        if critical > 0:
            logger.warning("\n⚠️  GRPO has CRITICAL issues that must be fixed!")
            logger.warning("The loss weighting is inverted - training will optimize in wrong direction!")
        else:
            logger.info("\n✓ No critical issues found")

        logger.info("\n" + "="*60)

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
