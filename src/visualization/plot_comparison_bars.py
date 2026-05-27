"""
Bar Chart Visualization with Statistical Significance Tests

Generates grouped bar charts comparing four policies — None, Passive Assistant,
Heuristic, MA-IPPO — with Mann-Whitney U tests for pairwise significance
against each baseline.

Usage:
    cd src/visualization
    python plot_comparison_bars.py [--results PATH] [--output-dir PATH]

Input:
    data/results/comparison_4policy.json

Output:
    results/figures/comparison_bars_reward.png
    results/figures/comparison_bars_failures.png
    results/figures/comparison_bars_comm.png
    results/figures/comparison_heatmap_improvement.png
    data/results/comparison_4policy_stats.json
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).parent.parent.parent

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Reuse constants from plot_comparison.py
FAIL_REGIMES = ['very_low', 'low', 'balanced', 'high', 'very_high']
FAIL_COSTS = {'very_low': 5, 'low': 10, 'balanced': 15, 'high': 30, 'very_high': 50}
OBS_NOISE_KEYS = ['low', 'normal', 'high']
OBS_NOISE_LABELS = {'low': 'Low Noise', 'normal': 'Normal Noise', 'high': 'High Noise'}

POLICY_COLORS = {
    'none':              '#8c8c8c',
    'passive_assistant': '#e69f00',
    'heuristic':         '#e8a735',
    'ma_ippo':           '#2e86de',
}
POLICY_LABELS = {
    'none':              'None',
    'passive_assistant': 'Passive Assistant',
    'heuristic':         'Heuristic',
    'ma_ippo':           'MA-IPPO (Ours)',
}

POLICIES_ALL = ['none', 'passive_assistant', 'heuristic', 'ma_ippo']
BASELINES = ['none', 'passive_assistant', 'heuristic']

# Number of pairwise comparisons for Bonferroni correction:
# 3 pairs (none/passive_assistant/heuristic vs IPPO) x 15 conditions
N_TOTAL_TESTS = len(BASELINES) * len(OBS_NOISE_KEYS) * len(FAIL_REGIMES)


# ============================================================================
# STATISTICAL TESTS
# ============================================================================
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled_std = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
                         / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def pairwise_test(
    episodes_a: List[Dict], episodes_b: List[Dict], metric: str = 'reward'
) -> Dict[str, Any]:
    """Mann-Whitney U test + Welch's t-test between two episode lists.

    Args:
        episodes_a: Per-episode dicts for policy A
        episodes_b: Per-episode dicts for policy B
        metric: Key to extract from each episode dict

    Returns:
        Dict with u_stat, u_pvalue, t_stat, t_pvalue, effect_size_d,
        bonferroni_u_pvalue, significance
    """
    a = np.array([ep[metric] for ep in episodes_a])
    b = np.array([ep[metric] for ep in episodes_b])

    # Mann-Whitney U (two-sided)
    u_stat, u_pvalue = sp_stats.mannwhitneyu(a, b, alternative='two-sided')

    # Welch's t-test
    t_stat, t_pvalue = sp_stats.ttest_ind(a, b, equal_var=False)

    d = cohens_d(a, b)

    # Bonferroni-corrected p-value
    bonf_p = min(u_pvalue * N_TOTAL_TESTS, 1.0)

    # Significance level (using Bonferroni-corrected p)
    if bonf_p < 0.001:
        sig = '***'
    elif bonf_p < 0.01:
        sig = '**'
    elif bonf_p < 0.05:
        sig = '*'
    else:
        sig = 'ns'

    return {
        'mean_a': float(np.mean(a)),
        'mean_b': float(np.mean(b)),
        'u_stat': float(u_stat),
        'u_pvalue': float(u_pvalue),
        'u_pvalue_bonf': float(bonf_p),
        't_stat': float(t_stat),
        't_pvalue': float(t_pvalue),
        'effect_size_d': float(d),
        'significance': sig,
        'n_a': len(a),
        'n_b': len(b),
    }


def run_all_tests(conditions: Dict) -> Dict[str, Any]:
    """Run pairwise significance tests for all conditions.

    Tests: passive vs MA-IPPO, heuristic vs MA-IPPO
    Metrics: reward, failures
    """
    all_results = {}

    for obs_key in OBS_NOISE_KEYS:
        for fail_regime in FAIL_REGIMES:
            cond_key = f"{obs_key}/{fail_regime}"
            all_results[cond_key] = {}

            for task_name, task_data in conditions.items():
                if obs_key not in task_data or fail_regime not in task_data[obs_key]:
                    continue
                cond = task_data[obs_key][fail_regime]

                ippo = cond.get('ma_ippo')
                if ippo is None or 'per_episode' not in ippo:
                    continue

                task_results = {}
                for baseline in BASELINES:
                    bl_data = cond.get(baseline)
                    if bl_data is None or 'per_episode' not in bl_data:
                        continue

                    pair_key = f"{baseline}_vs_ippo"
                    task_results[pair_key] = {}
                    for metric in ['reward', 'failures']:
                        task_results[pair_key][metric] = pairwise_test(
                            bl_data['per_episode'], ippo['per_episode'], metric
                        )

                if task_results:
                    all_results[cond_key][task_name] = task_results

    return all_results


def get_sig_marker(
    stats_results: Dict, obs_key: str, fail_regime: str,
    baseline: str, metric: str, task_name: str = 'make_cereal'
) -> str:
    """Get significance marker for a specific comparison."""
    cond_key = f"{obs_key}/{fail_regime}"
    try:
        return stats_results[cond_key][task_name][f"{baseline}_vs_ippo"][metric]['significance']
    except KeyError:
        return ''


# ============================================================================
# HELPER: Extract per-episode data for bar charts
# ============================================================================
def get_bar_data(
    conditions: Dict, task_name: str, obs_key: str, metric: str
) -> Dict[str, Dict[str, float]]:
    """Get mean and 95% CI for each policy across fail regimes.

    Returns:
        {policy: {'means': [5 values], 'ci95': [5 values]}}
    """
    policies = POLICIES_ALL
    result = {p: {'means': [], 'ci95': []} for p in policies}

    for fail_regime in FAIL_REGIMES:
        task_data = conditions.get(task_name, {})
        if obs_key not in task_data or fail_regime not in task_data[obs_key]:
            for p in policies:
                result[p]['means'].append(np.nan)
                result[p]['ci95'].append(0.0)
            continue

        cond = task_data[obs_key][fail_regime]

        for policy in policies:
            pol_data = cond.get(policy)
            if pol_data is None or 'per_episode' not in pol_data:
                result[policy]['means'].append(np.nan)
                result[policy]['ci95'].append(0.0)
                continue

            values = np.array([ep[metric] for ep in pol_data['per_episode']])
            mean = np.mean(values)
            ci95 = np.std(values, ddof=1) / np.sqrt(len(values)) * 1.96
            result[policy]['means'].append(mean)
            result[policy]['ci95'].append(ci95)

    return result


# ============================================================================
# FIGURE 1 & 2: Grouped Bar Charts (Reward / Failures)
# ============================================================================
def plot_grouped_bars(
    conditions: Dict, stats_results: Dict,
    metric: str, metric_label: str, output_path: Path,
    task_name: str = 'make_cereal',
):
    """Grouped bar chart: 3 panels (obs_noise) x 5 fail_regimes x 3 policies.

    Significance markers shown above bars for passive vs IPPO and heuristic vs IPPO.
    """
    policies = POLICIES_ALL
    n_regimes = len(FAIL_REGIMES)
    n_policies = len(policies)
    bar_width = 0.2
    x = np.arange(n_regimes)

    fig, axes = plt.subplots(len(OBS_NOISE_KEYS), 1, figsize=(10, 12), squeeze=False)

    for row, obs_key in enumerate(OBS_NOISE_KEYS):
        ax = axes[row, 0]
        data = get_bar_data(conditions, task_name, obs_key, metric)

        for i, policy in enumerate(policies):
            means = np.array(data[policy]['means'])
            ci95 = np.array(data[policy]['ci95'])
            offset = (i - 1.5) * bar_width
            bars = ax.bar(
                x + offset, means, bar_width,
                yerr=ci95, capsize=3,
                color=POLICY_COLORS[policy],
                label=POLICY_LABELS[policy],
                edgecolor='white', linewidth=0.5,
                error_kw={'linewidth': 1},
            )

        # Add significance markers
        for j, fail_regime in enumerate(FAIL_REGIMES):
            ippo_mean = data['ma_ippo']['means'][j]
            ippo_ci = data['ma_ippo']['ci95'][j]
            if np.isnan(ippo_mean):
                continue

            # Find max bar height in this group for positioning
            max_height = max(
                data[p]['means'][j] + data[p]['ci95'][j]
                for p in policies
                if not np.isnan(data[p]['means'][j])
            )

            for bi, baseline in enumerate(BASELINES):
                sig = get_sig_marker(stats_results, obs_key, fail_regime,
                                     baseline, metric, task_name)
                if sig and sig != 'ns':
                    y_offset = max_height * 0.03 + bi * max_height * 0.06
                    y_pos = max_height + y_offset + max_height * 0.02

                    # Draw bracket between baseline and IPPO bars
                    bl_idx = policies.index(baseline)
                    ippo_idx = policies.index('ma_ippo')
                    x_bl = j + (bl_idx - 1.5) * bar_width
                    x_ippo = j + (ippo_idx - 1.5) * bar_width
                    x_mid = (x_bl + x_ippo) / 2

                    bracket_y = y_pos
                    ax.plot([x_bl, x_bl, x_ippo, x_ippo],
                            [bracket_y - max_height * 0.01, bracket_y,
                             bracket_y, bracket_y - max_height * 0.01],
                            color='#333333', linewidth=0.8)
                    ax.text(x_mid, bracket_y + max_height * 0.005, sig,
                            ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([f"C_f={FAIL_COSTS[r]}" for r in FAIL_REGIMES], fontsize=9)
        ax.set_title(OBS_NOISE_LABELS[obs_key], fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_label, fontsize=11)
        ax.grid(True, alpha=0.2, axis='y')

        if row == 0:
            ax.legend(fontsize=10, loc='best')

    task_label = task_name.replace('_', ' ').title()
    fig.suptitle(f'{task_label}: {metric_label} by Condition',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()

    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# FIGURE 3: Communication Breakdown Stacked Bar
# ============================================================================
def plot_comm_bars(
    conditions: Dict, output_path: Path,
    task_name: str = 'make_cereal',
):
    """Stacked bar chart of MA-IPPO communication actions."""
    comm_metrics = ['narrations', 'questions', 'reminds', 'confirms']
    comm_labels = ['Narrations', 'Questions', 'Reminds', 'Confirms']
    comm_colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

    fig, axes = plt.subplots(1, len(OBS_NOISE_KEYS), figsize=(14, 5), squeeze=False)

    x = np.arange(len(FAIL_REGIMES))
    bar_width = 0.6

    for col, obs_key in enumerate(OBS_NOISE_KEYS):
        ax = axes[0, col]
        bottoms = np.zeros(len(FAIL_REGIMES))

        for metric, label, color in zip(comm_metrics, comm_labels, comm_colors):
            values = []
            ci_vals = []
            for fail_regime in FAIL_REGIMES:
                task_data = conditions.get(task_name, {})
                if obs_key not in task_data or fail_regime not in task_data[obs_key]:
                    values.append(0.0)
                    ci_vals.append(0.0)
                    continue
                cond = task_data[obs_key][fail_regime]
                ippo = cond.get('ma_ippo')
                if ippo is None or 'per_episode' not in ippo:
                    values.append(0.0)
                    ci_vals.append(0.0)
                    continue
                ep_vals = np.array([ep.get(metric, 0) for ep in ippo['per_episode']])
                values.append(float(np.mean(ep_vals)))
                ci_vals.append(float(np.std(ep_vals, ddof=1) / np.sqrt(len(ep_vals)) * 1.96))

            values = np.array(values)
            ax.bar(x, values, bar_width, bottom=bottoms, label=label,
                   color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
            bottoms += values

        ax.set_xticks(x)
        ax.set_xticklabels([f"C_f={FAIL_COSTS[r]}" for r in FAIL_REGIMES],
                           rotation=30, ha='right', fontsize=9)
        ax.set_title(OBS_NOISE_LABELS[obs_key], fontsize=12, fontweight='bold')
        if col == 0:
            ax.set_ylabel('Mean Count per Episode', fontsize=11)
        ax.grid(True, alpha=0.2, axis='y')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 1.08))

    task_label = task_name.replace('_', ' ').title()
    fig.suptitle(f'{task_label}: MA-IPPO Communication Breakdown',
                 fontsize=13, fontweight='bold', y=1.12)
    fig.tight_layout()

    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# FIGURE 4: Improvement Heatmap
# ============================================================================
def plot_improvement_heatmap(
    conditions: Dict, output_path: Path,
    task_name: str = 'make_cereal',
):
    """Heatmap of MA-IPPO reward improvement ratio over passive."""
    import pandas as pd

    matrix = []
    for obs_key in OBS_NOISE_KEYS:
        row = []
        for fail_regime in FAIL_REGIMES:
            task_data = conditions.get(task_name, {})
            if obs_key not in task_data or fail_regime not in task_data[obs_key]:
                row.append(np.nan)
                continue
            cond = task_data[obs_key][fail_regime]

            none_mean = cond.get('none', {}).get('mean_reward', np.nan)
            ippo_mean = cond.get('ma_ippo', {}).get('mean_reward', np.nan)

            if np.isnan(none_mean) or np.isnan(ippo_mean):
                row.append(np.nan)
            elif none_mean == 0:
                row.append(ippo_mean)
            else:
                # Improvement = (IPPO - None) / |None|
                row.append((ippo_mean - none_mean) / abs(none_mean))
        matrix.append(row)

    df = pd.DataFrame(
        matrix,
        index=[OBS_NOISE_LABELS[k] for k in OBS_NOISE_KEYS],
        columns=[f"C_f={FAIL_COSTS[r]}" for r in FAIL_REGIMES],
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        df, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
        linewidths=1, linecolor='white', ax=ax,
        cbar_kws={'label': 'Improvement Ratio'},
    )

    task_label = task_name.replace('_', ' ').title()
    ax.set_title(f'{task_label}: MA-IPPO Reward Improvement over None',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Observation Noise', fontsize=11)
    ax.set_xlabel('Failure Cost Regime', fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# RESULTS TABLE (stdout + JSON)
# ============================================================================
def print_stats_table(stats_results: Dict):
    """Print a formatted table of significance test results (one column per baseline)."""
    col_header_template = "{:^36}"
    sub_col = f"{'U-stat':>8}  {'p(bonf)':>8}  {'sig':>4}  {'d':>6}"
    header = f"{'Condition':<22} | " + " | ".join(
        col_header_template.format(f"{POLICY_LABELS[b]} vs IPPO") for b in BASELINES
    )
    sub = f"{'':22} | " + " | ".join(sub_col for _ in BASELINES)
    sep = '-' * len(sub)

    print(f"\n{sep}")
    print("Statistical Significance Tests (Mann-Whitney U, Bonferroni-corrected)")
    print(f"N total tests = {N_TOTAL_TESTS}")
    print(sep)
    print(header)
    print(sub)
    print(sep)

    for obs_key in OBS_NOISE_KEYS:
        for fail_regime in FAIL_REGIMES:
            cond_key = f"{obs_key}/{fail_regime}"
            cond_label = f"{obs_key}/C_f={FAIL_COSTS[fail_regime]}"

            cond_data = stats_results.get(cond_key, {})
            task_data = next(iter(cond_data.values()), {}) if cond_data else {}

            parts = [f"{cond_label:<22}"]
            for baseline in BASELINES:
                pair = task_data.get(f'{baseline}_vs_ippo', {}).get('reward', {})
                if pair:
                    u = pair['u_stat']
                    p = pair['u_pvalue_bonf']
                    sig = pair['significance']
                    d = pair['effect_size_d']
                    parts.append(f"{u:>8.0f}  {p:>8.4f}  {sig:>4}  {d:>6.3f}")
                else:
                    parts.append(f"{'N/A':>8}  {'N/A':>8}  {'N/A':>4}  {'N/A':>6}")

            print(f" | ".join(parts))

    print(sep)


# ============================================================================
# CROSS-TASK AVERAGE FIGURES
# ============================================================================
def _get_cross_task_bar_data(
    conditions: Dict, task_names: List[str], obs_key: str, metric: str
) -> Dict[str, Dict[str, float]]:
    """Get mean and 95% CI averaged across tasks for each policy."""
    policies = POLICIES_ALL
    result = {p: {'means': [], 'ci95': []} for p in policies}

    for fail_regime in FAIL_REGIMES:
        for policy in policies:
            all_values = []
            for task_name in task_names:
                task_data = conditions.get(task_name, {})
                if obs_key not in task_data or fail_regime not in task_data[obs_key]:
                    continue
                cond = task_data[obs_key][fail_regime]
                pol_data = cond.get(policy)
                if pol_data is None or 'per_episode' not in pol_data:
                    continue
                all_values.extend([ep[metric] for ep in pol_data['per_episode']])

            if all_values:
                arr = np.array(all_values)
                result[policy]['means'].append(float(np.mean(arr)))
                result[policy]['ci95'].append(
                    float(np.std(arr, ddof=1) / np.sqrt(len(arr)) * 1.96)
                )
            else:
                result[policy]['means'].append(np.nan)
                result[policy]['ci95'].append(0.0)

    return result


def _plot_cross_task_average(
    conditions: Dict, task_names: List[str],
    stats_results: Dict, output_dir: Path,
):
    """Plot cross-task average bar charts for reward and failures."""
    policies = POLICIES_ALL
    n_regimes = len(FAIL_REGIMES)
    bar_width = 0.2
    x = np.arange(n_regimes)

    for metric, metric_label in [('reward', 'Mean Reward'), ('failures', 'Mean Failures')]:
        fig, axes = plt.subplots(len(OBS_NOISE_KEYS), 1, figsize=(10, 12), squeeze=False)

        for row, obs_key in enumerate(OBS_NOISE_KEYS):
            ax = axes[row, 0]
            data = _get_cross_task_bar_data(conditions, task_names, obs_key, metric)

            for i, policy in enumerate(policies):
                means = np.array(data[policy]['means'])
                ci95 = np.array(data[policy]['ci95'])
                offset = (i - 1.5) * bar_width
                ax.bar(
                    x + offset, means, bar_width,
                    yerr=ci95, capsize=3,
                    color=POLICY_COLORS[policy],
                    label=POLICY_LABELS[policy],
                    edgecolor='white', linewidth=0.5,
                    error_kw={'linewidth': 1},
                )

            ax.set_xticks(x)
            ax.set_xticklabels([f"C_f={FAIL_COSTS[r]}" for r in FAIL_REGIMES], fontsize=9)
            ax.set_title(OBS_NOISE_LABELS[obs_key], fontsize=12, fontweight='bold')
            ax.set_ylabel(metric_label, fontsize=11)
            ax.grid(True, alpha=0.2, axis='y')

            if row == 0:
                ax.legend(fontsize=10, loc='best')

        n_tasks = len(task_names)
        fig.suptitle(f'Cross-Task Average ({n_tasks} tasks): {metric_label} by Condition',
                     fontsize=14, fontweight='bold')
        fig.tight_layout()

        out_path = output_dir / f"comparison_bars_{metric}_cross_task_avg.png"
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_path}")


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Bar chart comparison with statistical significance tests'
    )
    parser.add_argument(
        '--results', default=None,
        help='Path to comparison_4policy.json'
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='Output directory for figures'
    )
    parser.add_argument(
        '--stats-output', default=None,
        help='Output path for stats JSON'
    )
    args = parser.parse_args()

    results_path = Path(args.results) if args.results else (
        PROJECT_ROOT / "data" / "results" / "comparison_4policy.json"
    )
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "results" / "figures"
    )
    stats_output = Path(args.stats_output) if args.stats_output else (
        PROJECT_ROOT / "data" / "results" / "comparison_4policy_stats.json"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    conditions = results['conditions']
    task_names = sorted(conditions.keys())
    print(f"Loaded results: tasks={task_names}, "
          f"n_episodes={results.get('n_eval_episodes', '?')}")

    # Step 1: Run all statistical tests
    print("\nRunning pairwise significance tests...")
    stats_results = run_all_tests(conditions)

    # Step 2: Print results table
    print_stats_table(stats_results)

    # Step 3: Save stats JSON
    with open(stats_output, 'w') as f:
        json.dump(stats_results, f, indent=2)
    print(f"\nSaved stats: {stats_output}")

    # Step 4: Generate per-task figures
    print("\nGenerating figures...")

    for task_name in task_names:
        suffix = f"_{task_name}" if len(task_names) > 1 else ""
        print(f"\n  Task: {task_name}")

        plot_grouped_bars(
            conditions, stats_results,
            metric='reward', metric_label='Mean Reward',
            output_path=output_dir / f"comparison_bars_reward{suffix}.png",
            task_name=task_name,
        )

        plot_grouped_bars(
            conditions, stats_results,
            metric='failures', metric_label='Mean Failures',
            output_path=output_dir / f"comparison_bars_failures{suffix}.png",
            task_name=task_name,
        )

        plot_comm_bars(
            conditions, output_path=output_dir / f"comparison_bars_comm{suffix}.png",
            task_name=task_name,
        )

        plot_improvement_heatmap(
            conditions, output_path=output_dir / f"comparison_heatmap_improvement{suffix}.png",
            task_name=task_name,
        )

    # Step 5: Cross-task average figures (if multiple tasks)
    if len(task_names) > 1:
        print("\n  Generating cross-task average figures...")
        _plot_cross_task_average(conditions, task_names, stats_results, output_dir)

    print("\nAll figures generated.")


if __name__ == '__main__':
    main()
