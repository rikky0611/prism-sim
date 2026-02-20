"""
Cross-Task Multi-Regime RL Visualization

Generate comprehensive visualizations from evaluation results:
1. Performance heatmap (Task × Regime)
2. Policy comparison panels (3 regimes)
3. Interruptions heatmap
4. Failures heatmap
5. Summary dashboard (4-panel figure)

Usage:
    python visualize_cross_task_multi_regime.py
"""

import json
import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_evaluation_results() -> dict:
    """Load evaluation results from JSON file.

    Returns:
        Evaluation results dict
    """
    results_path = PROJECT_ROOT / "data" / "results" / "cross_task_multi_regime_evaluation.json"

    if not results_path.exists():
        raise FileNotFoundError(
            f"Evaluation results not found at {results_path}. "
            "Run evaluate_cross_task_multi_regime.py first."
        )

    with open(results_path, 'r') as f:
        return json.load(f)


def create_performance_heatmap(results: dict, save_path: Path):
    """Create task × regime performance heatmap with RL rewards.

    Args:
        results: Evaluation results dict
        save_path: Path to save figure
    """
    evaluation_data = results['results']
    regimes = list(evaluation_data.keys())
    tasks = list(evaluation_data[regimes[0]].keys())

    # Build matrices
    n_tasks = len(tasks)
    n_regimes = len(regimes)
    data = np.zeros((n_tasks, n_regimes))
    annotations = []

    for i, task in enumerate(tasks):
        row_annotations = []
        for j, regime in enumerate(regimes):
            task_data = evaluation_data[regime][task]
            rl_reward = task_data['policies']['RL_PPO'].get('mean_reward', 0)
            improvement = task_data.get('improvement_pct', 0)

            data[i, j] = rl_reward if rl_reward is not None else 0
            if rl_reward is not None and improvement is not None:
                row_annotations.append(f"{rl_reward:.1f}\n({improvement:+.1f}%)")
            else:
                row_annotations.append("N/A")
        annotations.append(row_annotations)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    # Format task and regime names
    task_labels = [t.replace('_', ' ').title() for t in tasks]
    regime_labels = [r.replace('_', ' ').title() for r in regimes]

    sns.heatmap(
        data,
        annot=np.array(annotations),
        fmt='',
        cmap='coolwarm_r',  # Reversed: higher (less negative) is better
        center=np.median(data[data != 0]) if np.any(data != 0) else -100,
        xticklabels=regime_labels,
        yticklabels=task_labels,
        cbar_kws={'label': 'RL Mean Reward'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )

    ax.set_title('RL Performance Across Tasks and Cost Regimes\n(Reward and Improvement %)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Cost Regime', fontsize=14, fontweight='bold')
    ax.set_ylabel('Task', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path.name}")


def create_policy_comparison_panels(results: dict, save_path: Path):
    """Create 3-panel policy comparison across regimes.

    Args:
        results: Evaluation results dict
        save_path: Path to save figure
    """
    evaluation_data = results['results']
    regimes = list(evaluation_data.keys())
    tasks = list(evaluation_data[regimes[0]].keys())
    policies = ['Random', 'Proactive', 'Reactive', 'RL_PPO']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Policy Comparison Across Cost Regimes', fontsize=18, fontweight='bold')

    colors = {'Random': '#E74C3C', 'Proactive': '#3498DB', 'Reactive': '#F39C12', 'RL_PPO': '#27AE60'}

    for idx, regime in enumerate(regimes):
        ax = axes[idx]

        # Collect mean rewards for each policy
        policy_rewards = {policy: [] for policy in policies}

        for task in tasks:
            task_data = evaluation_data[regime][task]
            for policy in policies:
                reward = task_data['policies'][policy].get('mean_reward')
                if reward is not None:
                    policy_rewards[policy].append(reward)

        # Compute means and stds
        means = [np.mean(policy_rewards[p]) if policy_rewards[p] else 0 for p in policies]
        stds = [np.std(policy_rewards[p]) if policy_rewards[p] else 0 for p in policies]

        # Create bar chart
        x = np.arange(len(policies))
        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=[colors[p] for p in policies],
                      alpha=0.8, edgecolor='black', linewidth=1.5)

        # Formatting
        regime_label = regime.replace('_', ' ').title()
        ax.set_title(f'{regime_label}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(policies, fontsize=11)
        ax.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.3)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (2 if height < 0 else -5),
                    f'{mean:.1f}',
                    ha='center', va='bottom' if height < 0 else 'top',
                    fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path.name}")


def create_interruptions_heatmap(results: dict, save_path: Path):
    """Create task × regime interruptions heatmap.

    Args:
        results: Evaluation results dict
        save_path: Path to save figure
    """
    evaluation_data = results['results']
    regimes = list(evaluation_data.keys())
    tasks = list(evaluation_data[regimes[0]].keys())

    # Build matrix
    n_tasks = len(tasks)
    n_regimes = len(regimes)
    data = np.zeros((n_tasks, n_regimes))

    for i, task in enumerate(tasks):
        for j, regime in enumerate(regimes):
            task_data = evaluation_data[regime][task]
            interruptions = task_data['policies']['RL_PPO'].get('mean_interruptions', 0)
            data[i, j] = interruptions if interruptions is not None else 0

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    task_labels = [t.replace('_', ' ').title() for t in tasks]
    regime_labels = [r.replace('_', ' ').title() for r in regimes]

    sns.heatmap(
        data,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        xticklabels=regime_labels,
        yticklabels=task_labels,
        cbar_kws={'label': 'Mean Interruptions'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )

    ax.set_title('RL Intervention Frequency Across Tasks and Regimes',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Cost Regime', fontsize=14, fontweight='bold')
    ax.set_ylabel('Task', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path.name}")


def create_failures_heatmap(results: dict, save_path: Path):
    """Create task × regime failures heatmap.

    Args:
        results: Evaluation results dict
        save_path: Path to save figure
    """
    evaluation_data = results['results']
    regimes = list(evaluation_data.keys())
    tasks = list(evaluation_data[regimes[0]].keys())

    # Build matrix
    n_tasks = len(tasks)
    n_regimes = len(regimes)
    data = np.zeros((n_tasks, n_regimes))

    for i, task in enumerate(tasks):
        for j, regime in enumerate(regimes):
            task_data = evaluation_data[regime][task]
            failures = task_data['policies']['RL_PPO'].get('mean_failures', 0)
            data[i, j] = failures if failures is not None else 0

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    task_labels = [t.replace('_', ' ').title() for t in tasks]
    regime_labels = [r.replace('_', ' ').title() for r in regimes]

    sns.heatmap(
        data,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',  # Red = high failures, Green = low failures
        xticklabels=regime_labels,
        yticklabels=task_labels,
        cbar_kws={'label': 'Mean Failures'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )

    ax.set_title('RL Failure Rates Across Tasks and Regimes',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Cost Regime', fontsize=14, fontweight='bold')
    ax.set_ylabel('Task', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path.name}")


def create_summary_dashboard(results: dict, save_path: Path):
    """Create comprehensive 4-panel summary dashboard.

    Args:
        results: Evaluation results dict
        save_path: Path to save figure
    """
    evaluation_data = results['results']
    aggregate_stats = results['aggregate_stats']
    regimes = list(evaluation_data.keys())
    tasks = list(evaluation_data[regimes[0]].keys())

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle('Cross-Task Multi-Regime RL Summary Dashboard',
                 fontsize=20, fontweight='bold', y=0.98)

    # ===== Panel 1: Improvement % by Regime (Box Plot) =====
    ax1 = fig.add_subplot(gs[0, 0])

    regime_improvements = {}
    for regime in regimes:
        improvements = []
        for task in tasks:
            task_data = evaluation_data[regime][task]
            improvement = task_data.get('improvement_pct')
            if improvement is not None:
                improvements.append(improvement)
        regime_improvements[regime] = improvements

    regime_labels = [r.replace('_', ' ').title() for r in regimes]
    bp = ax1.boxplot(
        regime_improvements.values(),
        labels=regime_labels,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='red', markersize=8)
    )

    for patch, color in zip(bp['boxes'], ['#E74C3C', '#3498DB', '#F39C12']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_title('Improvement % Distribution by Regime', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Improvement over Best Baseline (%)', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # ===== Panel 2: Improvement % by Task (Bar Chart) =====
    ax2 = fig.add_subplot(gs[0, 1])

    task_improvements = []
    for task in tasks:
        improvements = []
        for regime in regimes:
            task_data = evaluation_data[regime][task]
            improvement = task_data.get('improvement_pct')
            if improvement is not None:
                improvements.append(improvement)
        task_improvements.append(np.mean(improvements) if improvements else 0)

    task_labels = [t.replace('_', ' ').title() for t in tasks]
    colors = ['#27AE60' if imp > 0 else '#E74C3C' for imp in task_improvements]

    bars = ax2.barh(task_labels, task_improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_title('Mean Improvement % by Task', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Mean Improvement (%)', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels
    for bar, val in zip(bars, task_improvements):
        width = bar.get_width()
        ax2.text(width + (1 if width > 0 else -1), bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}%',
                 ha='left' if width > 0 else 'right',
                 va='center',
                 fontsize=9,
                 fontweight='bold')

    # ===== Panel 3: Interruptions vs Failures (Scatter) =====
    ax3 = fig.add_subplot(gs[1, 0])

    regime_colors = {'very_high_stakes': '#E74C3C', 'balanced': '#3498DB', 'moderate_low': '#F39C12'}

    for regime in regimes:
        interruptions_list = []
        failures_list = []

        for task in tasks:
            task_data = evaluation_data[regime][task]
            rl_data = task_data['policies']['RL_PPO']
            interruptions = rl_data.get('mean_interruptions')
            failures = rl_data.get('mean_failures')

            if interruptions is not None and failures is not None:
                interruptions_list.append(interruptions)
                failures_list.append(failures)

        regime_label = regime.replace('_', ' ').title()
        ax3.scatter(interruptions_list, failures_list,
                   label=regime_label,
                   color=regime_colors[regime],
                   s=100,
                   alpha=0.7,
                   edgecolor='black',
                   linewidth=1)

    ax3.set_title('Interruptions vs Failures Trade-off', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Mean Interruptions', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Mean Failures', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10, loc='best')
    ax3.grid(alpha=0.3, linestyle='--')

    # ===== Panel 4: Overall Statistics (Text Summary) =====
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    overall_stats = aggregate_stats['overall']

    summary_text = f"""
    OVERALL PERFORMANCE SUMMARY
    {'='*50}

    Mean Improvement:      {overall_stats['mean_improvement_pct']:.2f}%
    Median Improvement:    {overall_stats['median_improvement_pct']:.2f}%
    Std Improvement:       {overall_stats['std_improvement_pct']:.2f}%

    Success Rate:          {overall_stats['success_rate']*100:.1f}%
    Models Improved:       {overall_stats['n_models_improved']}/{overall_stats['n_models_evaluated']}

    {'='*50}
    BEST CASE
    {'='*50}
    """

    if overall_stats['best_case']:
        best = overall_stats['best_case']
        summary_text += f"""
    Regime:  {best['regime'].replace('_', ' ').title()}
    Task:    {best['task'].replace('_', ' ').title()}
    Improvement:  {best['improvement_pct']:.2f}%
    Reward:       {best['rl_reward']:.2f}
        """

    summary_text += f"""
    {'='*50}
    WORST CASE
    {'='*50}
    """

    if overall_stats['worst_case']:
        worst = overall_stats['worst_case']
        summary_text += f"""
    Regime:  {worst['regime'].replace('_', ' ').title()}
    Task:    {worst['task'].replace('_', ' ').title()}
    Improvement:  {worst['improvement_pct']:.2f}%
    Reward:       {worst['rl_reward']:.2f}
        """

    ax4.text(0.05, 0.95, summary_text,
             transform=ax4.transAxes,
             fontsize=11,
             verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path.name}")


def generate_all_visualizations():
    """Generate all visualization figures."""
    print("="*70)
    print("CROSS-TASK MULTI-REGIME RL VISUALIZATION")
    print("="*70)
    print("Loading evaluation results...")

    results = load_evaluation_results()

    figures_dir = PROJECT_ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating visualizations in: {figures_dir}")
    print()

    # Generate all figures
    create_performance_heatmap(
        results,
        figures_dir / "cross_task_performance_heatmap.png"
    )

    create_policy_comparison_panels(
        results,
        figures_dir / "cross_task_policy_comparison.png"
    )

    create_interruptions_heatmap(
        results,
        figures_dir / "cross_task_interruptions_heatmap.png"
    )

    create_failures_heatmap(
        results,
        figures_dir / "cross_task_failures_heatmap.png"
    )

    create_summary_dashboard(
        results,
        figures_dir / "cross_task_summary_dashboard.png"
    )

    print()
    print("="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"Generated 5 figures (PNG + PDF) in: {figures_dir}")
    print("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate visualizations from cross-task multi-regime evaluation'
    )
    args = parser.parse_args()

    generate_all_visualizations()


if __name__ == '__main__':
    main()
