"""
Visualization and analysis for Task 2 results
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns

logger = logging.getLogger(__name__)


class ResultsVisualizer:
    """Visualize and analyze alignment results"""

    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")

    def plot_training_curves(
        self,
        metrics_history: Dict[str, List[float]],
        title: str = "Training Metrics",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot training metrics over time"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Loss curves
        if 'train_loss' in metrics_history or any('loss' in k for k in metrics_history.keys()):
            ax = axes[0, 0]
            for key in metrics_history.keys():
                if 'loss' in key.lower():
                    ax.plot(metrics_history[key], label=key, marker='o')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Reward curves
        if any('reward' in k for k in metrics_history.keys()):
            ax = axes[0, 1]
            for key in metrics_history.keys():
                if 'reward' in key.lower():
                    ax.plot(metrics_history[key], label=key, marker='s')
            ax.set_xlabel('Epoch/Step')
            ax.set_ylabel('Reward')
            ax.set_title('Reward Progression')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Metrics summary
        ax = axes[1, 0]
        ax.axis('off')
        summary_text = "Training Summary:\n"
        for key, values in metrics_history.items():
            if isinstance(values, list) and len(values) > 0:
                if isinstance(values[0], (int, float)):
                    summary_text += f"{key}: {values[-1]:.4f}\n"
        ax.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')

        # Info
        ax = axes[1, 1]
        ax.axis('off')

        plt.tight_layout()
        if save_path is None:
            save_path = self.output_dir / "training_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curve saved to {save_path}")
        plt.close()

    def plot_verbosity_analysis(
        self,
        length_stats: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
    ) -> None:
        """Plot verbosity analysis comparing methods"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Verbosity Analysis Across Methods', fontsize=16, fontweight='bold')

        methods = list(length_stats.keys())
        means = [length_stats[m].get('mean_length', 0) for m in methods]
        stds = [length_stats[m].get('std_length', 0) for m in methods]
        medians = [length_stats[m].get('median_length', 0) for m in methods]

        # Mean vs Std
        ax = axes[0, 0]
        ax.scatter(means, stds, s=200, alpha=0.6)
        for i, method in enumerate(methods):
            ax.annotate(method, (means[i], stds[i]), fontsize=10)
        ax.set_xlabel('Mean Response Length')
        ax.set_ylabel('Std Dev')
        ax.set_title('Mean vs Variability')
        ax.grid(True, alpha=0.3)

        # Bar plot: Mean lengths
        ax = axes[0, 1]
        ax.bar(methods, means, color='skyblue', alpha=0.7, label='Mean')
        ax.errorbar(methods, means, yerr=stds, fmt='none', color='black', capsize=5)
        ax.set_ylabel('Length (words)')
        ax.set_title('Mean Response Length ± Std')
        ax.grid(True, alpha=0.3, axis='y')

        # Comparison table
        ax = axes[1, 0]
        ax.axis('off')
        table_data = []
        for method in methods:
            stats = length_stats[method]
            table_data.append([
                method,
                f"{stats.get('mean_length', 0):.1f}",
                f"{stats.get('std_length', 0):.1f}",
                f"{stats.get('median_length', 0):.1f}",
            ])

        table = ax.table(
            cellText=table_data,
            colLabels=['Method', 'Mean', 'Std', 'Median'],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax.set_title('Length Statistics', fontweight='bold')

        # Distribution comparison
        ax = axes[1, 1]
        for method, stats in length_stats.items():
            ax.text(0.05, 0.9 - list(length_stats.keys()).index(method) * 0.15,
                   f"{method}: μ={stats.get('mean_length', 0):.1f}, "
                   f"σ={stats.get('std_length', 0):.1f}",
                   fontsize=11, transform=ax.transAxes)
        ax.axis('off')

        plt.tight_layout()
        if save_path is None:
            save_path = self.output_dir / "verbosity_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Verbosity analysis saved to {save_path}")
        plt.close()

    def plot_alignment_comparison(
        self,
        alignment_metrics: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
    ) -> None:
        """Compare alignment methods on key metrics"""
        methods = list(alignment_metrics.keys())
        metrics_names = list(alignment_metrics[methods[0]].keys())

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Alignment Methods Comparison', fontsize=16, fontweight='bold')

        # KL divergence comparison
        ax = axes[0, 0]
        kl_values = [alignment_metrics[m].get('kl_divergence', 0) for m in methods]
        colors = ['green' if kl < np.mean(kl_values) else 'red' for kl in kl_values]
        ax.bar(methods, kl_values, color=colors, alpha=0.7)
        ax.set_ylabel('KL Divergence')
        ax.set_title('Drift from Original Model (Lower is Better)')
        ax.grid(True, alpha=0.3, axis='y')

        # Perplexity comparison
        ax = axes[0, 1]
        ppl_values = [alignment_metrics[m].get('perplexity', 0) for m in methods]
        colors = ['green' if ppl < np.mean(ppl_values) else 'red' for ppl in ppl_values]
        ax.bar(methods, ppl_values, color=colors, alpha=0.7)
        ax.set_ylabel('Perplexity')
        ax.set_title('Instruction Following Capability (Lower is Better)')
        ax.grid(True, alpha=0.3, axis='y')

        # Metric summary table
        ax = axes[1, 0]
        ax.axis('off')
        table_data = []
        for method in methods:
            table_data.append([
                method,
                f"{alignment_metrics[method].get('kl_divergence', 0):.4f}",
                f"{alignment_metrics[method].get('perplexity', 0):.2f}",
            ])

        table = ax.table(
            cellText=table_data,
            colLabels=['Method', 'KL Div', 'Perplexity'],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Trade-off analysis
        ax = axes[1, 1]
        kl_norm = np.array(kl_values) / np.max(kl_values) if np.max(kl_values) > 0 else np.array(kl_values)
        ppl_norm = np.array(ppl_values) / np.max(ppl_values) if np.max(ppl_values) > 0 else np.array(ppl_values)

        for i, method in enumerate(methods):
            ax.scatter(kl_norm[i], ppl_norm[i], s=300, alpha=0.6, label=method)
            ax.annotate(method, (kl_norm[i], ppl_norm[i]), fontsize=10)

        ax.set_xlabel('Normalized KL Divergence')
        ax.set_ylabel('Normalized Perplexity')
        ax.set_title('Alignment Trade-off (Lower is Better)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        if save_path is None:
            save_path = self.output_dir / "alignment_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Alignment comparison saved to {save_path}")
        plt.close()

    def save_sample_generations(
        self,
        generations: Dict[str, Dict[str, List[str]]],
        save_path: Optional[str] = None,
    ) -> None:
        """Save generated samples to file for inspection"""
        if save_path is None:
            save_path = self.output_dir / "sample_generations.json"

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(generations, f, indent=2, ensure_ascii=False)

        logger.info(f"Sample generations saved to {save_path}")

    def create_summary_report(
        self,
        all_metrics: Dict,
        save_path: Optional[str] = None,
    ) -> None:
        """Create comprehensive summary report"""
        if save_path is None:
            save_path = self.output_dir / "summary_report.json"

        summary = {
            "alignment_methods": list(all_metrics.get('alignment_comparison', {}).keys()),
            "metrics_computed": {
                "catastrophic_forgetting": "kl_divergence, perplexity",
                "verbosity_bias": "mean_length, std_length, skewness",
                "reward_hacking": "reward_gap, hacking_rate",
            },
            "results": all_metrics,
        }

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Summary report saved to {save_path}")
