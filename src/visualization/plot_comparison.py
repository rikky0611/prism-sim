"""
Visualization for Comparison Experiment: Passive vs Heuristic vs MA-IPPO

Generates:
1. Main figure: 3x3 panels (metric x obs_noise), averaged across tasks
2. Per-task supplementary figures
3. Communication breakdown stacked bars for MA-IPPO

Usage:
    cd src/visualization
    python plot_comparison.py [--results PATH] [--output-dir PATH]

Input:
    data/results/comparison_3policy.json

Output:
    results/figures/comparison_main.png
    results/figures/comparison_per_task_<task>.png
    results/figures/comparison_comm_breakdown.png
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ============================================================================
# CONSTANTS
# ============================================================================
FAIL_REGIMES = ['very_low', 'low', 'balanced', 'high', 'very_high']
FAIL_COSTS = {'very_low': 5, 'low': 10, 'balanced': 15, 'high': 30, 'very_high': 50}
OBS_NOISE_KEYS = ['low', 'normal', 'high']
OBS_NOISE_LABELS = {'low': 'Low Noise', 'normal': 'Normal Noise', 'high': 'High Noise'}

POLICY_COLORS = {
    'passive': '#8c8c8c',
    'heuristic': '#e8a735',
    'ma_ippo': '#2e86de',
}
POLICY_LABELS = {
    'passive': 'Passive',
    'heuristic': 'Heuristic',
    'ma_ippo': 'MA-IPPO (Ours)',
}
POLICY_MARKERS = {
    'passive': 's',
    'heuristic': '^',
    'ma_ippo': 'o',
}


# ============================================================================
# DATA EXTRACTION
# ============================================================================
def extract_metric_across_tasks(
    conditions: Dict,
    obs_key: str,
    metric: str,
    policies: List[str] = None,
) -> Dict[str, Dict[str, List[float]]]:
    """Extract a metric for all fail regimes, averaged across tasks.

    Returns:
        {policy: {'means': [...], 'stds': [...]}} for each fail regime
    """
    if policies is None:
        policies = ['passive', 'heuristic', 'ma_ippo']

    result = {p: {'means': [], 'stds': []} for p in policies}

    for fail_regime in FAIL_REGIMES:
        task_values = {p: [] for p in policies}

        for task_name, task_data in conditions.items():
            if obs_key not in task_data or fail_regime not in task_data[obs_key]:
                continue
            cond = task_data[obs_key][fail_regime]

            for policy in policies:
                if cond.get(policy) is not None and metric in cond[policy]:
                    task_values[policy].append(cond[policy][metric])

        for policy in policies:
            vals = task_values[policy]
            if vals:
                result[policy]['means'].append(np.mean(vals))
                result[policy]['stds'].append(np.std(vals) / max(np.sqrt(len(vals)), 1))
            else:
                result[policy]['means'].append(np.nan)
                result[policy]['stds'].append(0.0)

    return result


def extract_metric_single_task(
    conditions: Dict,
    task_name: str,
    obs_key: str,
    metric: str,
    policies: List[str] = None,
) -> Dict[str, Dict[str, List[float]]]:
    """Extract a metric for a single task across fail regimes."""
    if policies is None:
        policies = ['passive', 'heuristic', 'ma_ippo']

    result = {p: {'means': [], 'stds': []} for p in policies}

    if task_name not in conditions:
        return result
    task_data = conditions[task_name]

    for fail_regime in FAIL_REGIMES:
        if obs_key not in task_data or fail_regime not in task_data[obs_key]:
            for p in policies:
                result[p]['means'].append(np.nan)
                result[p]['stds'].append(0.0)
            continue

        cond = task_data[obs_key][fail_regime]
        for policy in policies:
            if cond.get(policy) is not None and metric in cond[policy]:
                result[policy]['means'].append(cond[policy][metric])
                # Use std_reward for reward, 0 for other metrics
                if metric == 'mean_reward' and 'std_reward' in cond[policy]:
                    result[policy]['stds'].append(cond[policy]['std_reward'])
                else:
                    result[policy]['stds'].append(0.0)
            else:
                result[policy]['means'].append(np.nan)
                result[policy]['stds'].append(0.0)

    return result


# ============================================================================
# MAIN FIGURE: AVERAGED ACROSS TASKS
# ============================================================================
def plot_main_figure(conditions: Dict, output_dir: Path):
    """Generate the main comparison figure for the paper.

    Layout: 3 rows (metrics) x 3 columns (obs_noise levels)
    Metrics: reward, failures, reminds
    """
    metrics = [
        ('mean_reward', 'Mean Reward'),
        ('mean_failures', 'Mean Failures'),
        ('mean_reminds', 'Mean Reminds'),
    ]

    fig, axes = plt.subplots(len(metrics), len(OBS_NOISE_KEYS),
                             figsize=(14, 10), squeeze=False)

    x = np.array([FAIL_COSTS[r] for r in FAIL_REGIMES])

    for col, obs_key in enumerate(OBS_NOISE_KEYS):
        for row, (metric, metric_label) in enumerate(metrics):
            ax = axes[row, col]
            data = extract_metric_across_tasks(conditions, obs_key, metric)

            for policy in ['passive', 'heuristic', 'ma_ippo']:
                means = np.array(data[policy]['means'])
                stds = np.array(data[policy]['stds'])

                ax.plot(x, means,
                        color=POLICY_COLORS[policy],
                        marker=POLICY_MARKERS[policy],
                        label=POLICY_LABELS[policy],
                        linewidth=2, markersize=6)
                ax.fill_between(x, means - stds, means + stds,
                                color=POLICY_COLORS[policy], alpha=0.15)

            ax.set_xscale('log')
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in x])
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

            if row == 0:
                ax.set_title(OBS_NOISE_LABELS[obs_key], fontsize=13, fontweight='bold')
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=11)
            if row == len(metrics) - 1:
                ax.set_xlabel('Failure Cost Scale', fontsize=10)

            ax.grid(True, alpha=0.3)

    # Single legend at top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, 1.02))

    fig.suptitle('Passive vs Heuristic vs MA-IPPO (Averaged Across Tasks)',
                 fontsize=14, fontweight='bold', y=1.06)
    fig.tight_layout()

    out_path = output_dir / "comparison_main.png"
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================================
# PER-TASK SUPPLEMENTARY FIGURES
# ============================================================================
def plot_per_task(conditions: Dict, output_dir: Path):
    """Generate per-task comparison figures."""
    metrics = [
        ('mean_reward', 'Mean Reward'),
        ('mean_failures', 'Mean Failures'),
        ('mean_reminds', 'Mean Reminds'),
    ]

    x = np.array([FAIL_COSTS[r] for r in FAIL_REGIMES])

    for task_name in conditions:
        fig, axes = plt.subplots(len(metrics), len(OBS_NOISE_KEYS),
                                 figsize=(14, 10), squeeze=False)

        for col, obs_key in enumerate(OBS_NOISE_KEYS):
            for row, (metric, metric_label) in enumerate(metrics):
                ax = axes[row, col]
                data = extract_metric_single_task(conditions, task_name, obs_key, metric)

                for policy in ['passive', 'heuristic', 'ma_ippo']:
                    means = np.array(data[policy]['means'])
                    stds = np.array(data[policy]['stds'])

                    ax.plot(x, means,
                            color=POLICY_COLORS[policy],
                            marker=POLICY_MARKERS[policy],
                            label=POLICY_LABELS[policy],
                            linewidth=2, markersize=6)
                    if np.any(stds > 0):
                        ax.fill_between(x, means - stds, means + stds,
                                        color=POLICY_COLORS[policy], alpha=0.15)

                ax.set_xscale('log')
                ax.set_xticks(x)
                ax.set_xticklabels([str(v) for v in x])
                ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

                if row == 0:
                    ax.set_title(OBS_NOISE_LABELS[obs_key], fontsize=13, fontweight='bold')
                if col == 0:
                    ax.set_ylabel(metric_label, fontsize=11)
                if row == len(metrics) - 1:
                    ax.set_xlabel('Failure Cost Scale', fontsize=10)

                ax.grid(True, alpha=0.3)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=11,
                   bbox_to_anchor=(0.5, 1.02))

        task_label = task_name.replace('_', ' ').title()
        fig.suptitle(f'{task_label}: Passive vs Heuristic vs MA-IPPO',
                     fontsize=14, fontweight='bold', y=1.06)
        fig.tight_layout()

        out_path = output_dir / f"comparison_per_task_{task_name}.png"
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_path}")


# ============================================================================
# COMMUNICATION BREAKDOWN
# ============================================================================
def plot_comm_breakdown(conditions: Dict, output_dir: Path):
    """Stacked bar chart of MA-IPPO communication actions across conditions.

    Shows narrations, questions, reminds, confirms for MA-IPPO.
    One subplot per obs_noise level; bars grouped by fail_regime, averaged across tasks.
    """
    comm_metrics = ['mean_narrations', 'mean_questions', 'mean_reminds', 'mean_confirms']
    comm_labels = ['Narrations', 'Questions', 'Reminds', 'Confirms']
    comm_colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

    fig, axes = plt.subplots(1, len(OBS_NOISE_KEYS), figsize=(15, 5), squeeze=False)

    x = np.arange(len(FAIL_REGIMES))
    bar_width = 0.6

    for col, obs_key in enumerate(OBS_NOISE_KEYS):
        ax = axes[0, col]

        bottoms = np.zeros(len(FAIL_REGIMES))

        for m_idx, (metric, label, color) in enumerate(
            zip(comm_metrics, comm_labels, comm_colors)
        ):
            values = []
            for fail_regime in FAIL_REGIMES:
                task_vals = []
                for task_name, task_data in conditions.items():
                    if obs_key not in task_data or fail_regime not in task_data[obs_key]:
                        continue
                    cond = task_data[obs_key][fail_regime]
                    if cond.get('ma_ippo') is not None and metric in cond['ma_ippo']:
                        task_vals.append(cond['ma_ippo'][metric])
                values.append(np.mean(task_vals) if task_vals else 0.0)

            values = np.array(values)
            ax.bar(x, values, bar_width, bottom=bottoms, label=label, color=color, alpha=0.85)
            bottoms += values

        ax.set_xticks(x)
        ax.set_xticklabels(FAIL_REGIMES, rotation=30, ha='right', fontsize=9)
        ax.set_title(OBS_NOISE_LABELS[obs_key], fontsize=12, fontweight='bold')
        if col == 0:
            ax.set_ylabel('Mean Count per Episode', fontsize=11)
        ax.grid(True, alpha=0.2, axis='y')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 1.08))

    fig.suptitle('MA-IPPO Communication Breakdown (Averaged Across Tasks)',
                 fontsize=13, fontweight='bold', y=1.12)
    fig.tight_layout()

    out_path = output_dir / "comparison_comm_breakdown.png"
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================================
# CLI
# ============================================================================
def main():
    if not HAS_MPL:
        print("ERROR: matplotlib is required. Install with: pip install matplotlib")
        sys.exit(1)

    parser = argparse.ArgumentParser(description='Plot comparison experiment results')
    parser.add_argument(
        '--results', default=None,
        help='Path to comparison_3policy.json (default: data/results/comparison_3policy.json)'
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='Output directory for figures (default: results/figures/)'
    )
    args = parser.parse_args()

    results_path = Path(args.results) if args.results else (
        PROJECT_ROOT / "data" / "results" / "comparison_3policy.json"
    )
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "results" / "figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        print("Run 'python run_comparison_experiment.py' first.")
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    conditions = results['conditions']
    print(f"Loaded results for {len(conditions)} tasks")

    plot_main_figure(conditions, output_dir)
    plot_per_task(conditions, output_dir)
    plot_comm_breakdown(conditions, output_dir)

    print("\nAll figures generated.")


if __name__ == '__main__':
    main()
