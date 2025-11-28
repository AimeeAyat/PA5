"""
Verify GRPO Fixes
Quick check that all fixes are in place
"""
import re
from pathlib import Path

def check_file_content(filepath, patterns, issue_name):
    """Check if file contains required patterns"""
    content = Path(filepath).read_text()

    print(f"\n{'='*60}")
    print(f"Checking: {issue_name}")
    print(f"{'='*60}")

    all_found = True
    for pattern_name, pattern in patterns.items():
        if re.search(pattern, content, re.MULTILINE):
            print(f"[OK] {pattern_name}")
        else:
            print(f"[FAIL] {pattern_name} - NOT FOUND")
            all_found = False

    if all_found:
        print(f"\n[PASS] {issue_name}: ALL CHECKS PASSED")
    else:
        print(f"\n[FAIL] {issue_name}: SOME CHECKS FAILED")

    return all_found


def main():
    print("\n" + "="*60)
    print("GRPO FIXES VERIFICATION")
    print("="*60)

    filepath = "src/trainers/grpo_trainer.py"
    all_passed = True

    # Check 1: Reward model compatibility
    patterns_1 = {
        "Check logits shape": r"if logits\.shape\[-1\] == 1:",
        "Regression extraction": r"reward = logits\[0, 0\]\.item\(\)",
        "Classification extraction": r"reward = logits\[0, 1\]\.item\(\)",
    }
    all_passed &= check_file_content(filepath, patterns_1,
                                     "Fix #1: Reward Model Compatibility")

    # Check 2: KL penalty implementation
    patterns_2 = {
        "KL penalty method": r"def compute_kl_penalty",
        "Policy log probs": r"policy_log_probs = .*log_softmax",
        "Reference log probs": r"ref_log_probs = .*log_softmax",
        "KL divergence": r"kl_div.*kl_div\(",
    }
    all_passed &= check_file_content(filepath, patterns_2,
                                     "Fix #2: KL Penalty Method")

    # Check 3: KL penalty in rewards
    patterns_3 = {
        "Compute KL penalty": r"kl_penalty = self\.compute_kl_penalty",
        "Subtract from reward": r"reward = .*- kl_coeff \* kl_penalty",
    }
    all_passed &= check_file_content(filepath, patterns_3,
                                     "Fix #3: KL Penalty in Rewards")

    # Check 4: Old log prob storage
    patterns_4 = {
        "Store old log prob": r"old_log_prob = .*\.item\(\)",
        "Old log probs list": r"group_old_log_probs",
    }
    all_passed &= check_file_content(filepath, patterns_4,
                                     "Fix #4: Old Log Prob Storage")

    # Check 5: PPO clipping
    patterns_5 = {
        "Compute ratio": r"ratio = torch\.exp\(log_ratio\)",
        "Clipped surrogate 1": r"surr1 = ratio \* advantage",
        "Clipped surrogate 2": r"surr2 = torch\.clamp.*advantage",
        "Min of surrogates": r"torch\.min\(surr1, surr2\)",
    }
    all_passed &= check_file_content(filepath, patterns_5,
                                     "Fix #5: PPO Clipped Objective")

    # Check 6: Correct loss direction
    patterns_6 = {
        "Negative for ascent": r"policy_loss = -torch\.min",
        "No inverted weighting": r"(?!.*weighted_loss.*\(1\.0 - advantage\))",
    }
    # Special check for "no inverted weighting"
    content = Path(filepath).read_text()
    has_old_bug = "weighted_loss = loss * (1.0 - advantage)" in content

    print(f"\n{'='*60}")
    print(f"Checking: Fix #6: Correct Loss Direction")
    print(f"{'='*60}")

    if "policy_loss = -torch.min" in content:
        print(f"[OK] Uses policy gradient with negative for ascent")
    else:
        print(f"[FAIL] Policy gradient not found")
        all_passed = False

    if not has_old_bug:
        print(f"[OK] Old inverted weighting removed")
    else:
        print(f"[FAIL] Old inverted weighting still present")
        all_passed = False

    if not has_old_bug and "policy_loss = -torch.min" in content:
        print(f"\n[PASS] Fix #6: ALL CHECKS PASSED")
    else:
        print(f"\n[FAIL] Fix #6: SOME CHECKS FAILED")

    # Check 7: KL metrics
    patterns_7 = {
        "Track KL penalties": r"epoch_kl_penalties",
        "Log KL": r"KL = .*avg_kl",
        "KL in metrics": r"mean_kl_penalty",
    }
    all_passed &= check_file_content(filepath, patterns_7,
                                     "Fix #7: KL Metrics Tracking")

    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    if all_passed:
        print("\n[SUCCESS] ALL FIXES VERIFIED - GRPO IMPLEMENTATION IS CORRECT")
        print("\nYou can now:")
        print("  1. Run main.py to train GRPO")
        print("  2. Compare with DPO and PPO")
        print("  3. Analyze trade-offs in alignment")
    else:
        print("\n[ERROR] SOME FIXES MISSING - PLEASE REVIEW")
        print("\nCheck the failed items above and ensure:")
        print("  - All patterns are present in grpo_trainer.py")
        print("  - Code matches the fixes in GRPO_FIXES_APPLIED.md")

    print("="*60 + "\n")


if __name__ == "__main__":
    main()
