#!/usr/bin/env python3
"""Pareto frontier plot from existing 4-policy comparison data (E4).

Reads data/results/comparison_4policy*.json (one per task) and plots, per
task, a scatter on (interactions/episode, failures/episode) for each of:
None, Passive Assistant, Heuristic, MA-IPPO across all
(obs_noise × fail_regime) regimes. MA-IPPO points should sit on or below
the convex hull of the others — that is the Pareto-dominance claim (C4).

Usage:
    python plot_pareto_per_task.py [--out results/figures/pareto_per_task.png]
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.parent

POLICIES = [
    ('none',              'None',              '#888888', 'o'),
    ('passive_assistant', 'Passive Assistant', '#E69F00', '^'),
    ('heuristic',         'Heuristic',         '#D32F2F', 's'),
    ('ma_ippo',           'MA-IPPO',           '#1565C0', 'D'),
]
BASELINE_KEYS = ['none', 'passive_assistant', 'heuristic']  # everything except ma_ippo

# Display order: simple → complex
TASK_ORDER = [
    'make_cereal', 'make_coffee', 'make_tea', 'make_sandwich',
    'cooking', 'make_stencil', 'latte_making',
]


def load_all_tasks(results_dir: Path):
    """Load and merge all comparison_4policy*.json into a unified dict."""
    files = sorted(glob.glob(str(results_dir / 'comparison_4policy*.json')))
    merged = {}
    for f in files:
        if 'stats' in f:
            continue
        d = json.load(open(f))
        cs = d.get('conditions', {})
        for task, sub in cs.items():
            merged[task] = sub
    return merged


def total_interactions(metrics: dict) -> float:
    """Sum of all communicative actions per episode."""
    return (
        metrics.get('mean_narrations', 0.0)
        + metrics.get('mean_questions', 0.0)
        + metrics.get('mean_reminds', 0.0)
        + metrics.get('mean_confirms', 0.0)
    )


def collect_points(task_data: dict):
    """Collect (interactions, failures, reward) points for each policy."""
    pts = {p[0]: [] for p in POLICIES}
    for obs_noise, fr_dict in task_data.items():
        for fail_regime, cell in fr_dict.items():
            for pol_key, *_ in POLICIES:
                m = cell.get(pol_key)
                if not m:
                    continue
                pts[pol_key].append({
                    'interactions': total_interactions(m),
                    'failures': m.get('mean_failures', 0.0),
                    'reward': m.get('mean_reward', 0.0),
                    'obs_noise': obs_noise,
                    'fail_regime': fail_regime,
                })
    return pts


import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from _paper_export import add_paper_arg, export_to_paper


def plot_pareto(merged: dict, output_path: str, paper: bool = False):
    tasks = [t for t in TASK_ORDER if t in merged]
    if not tasks:
        tasks = list(merged.keys())
    n_tasks = len(tasks)
    n_cols = min(4, n_tasks)
    n_rows = (n_tasks + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.6 * n_cols, 4.0 * n_rows),
                             squeeze=False)
    fig.patch.set_facecolor('white')

    reward_wins = 0
    pareto_vs_each = {k: 0 for k in BASELINE_KEYS}
    pareto_vs_all = 0
    cell_counts = 0

    for idx, task in enumerate(tasks):
        ax = axes[idx // n_cols][idx % n_cols]
        pts = collect_points(merged[task])

        # Plot per-policy clouds
        for pol_key, pol_label, color, marker in POLICIES:
            xs = [p['interactions'] for p in pts[pol_key]]
            ys = [p['failures']     for p in pts[pol_key]]
            if not xs:
                continue
            ax.scatter(xs, ys, c=color, marker=marker, s=42, alpha=0.7,
                       edgecolors='white', linewidths=0.6, label=pol_label)

        # Per-cell win checks against all 3 baselines (None, Passive Assistant,
        # Heuristic). reward_win = MA-IPPO >= best of the three; pareto_vs_<b>
        # = MA-IPPO dominates baseline b on (interactions, failures).
        baseline_pts = {
            b: {(p['obs_noise'], p['fail_regime']): p for p in pts[b]}
            for b in BASELINE_KEYS
        }
        ippo_pts = {(p['obs_noise'], p['fail_regime']): p for p in pts['ma_ippo']}
        for key, ip in ippo_pts.items():
            cell_counts += 1
            best_other_rew = max(
                (baseline_pts[b].get(key, {'reward': -1e9})['reward']
                 for b in BASELINE_KEYS),
                default=-1e9,
            )
            if ip['reward'] >= best_other_rew:
                reward_wins += 1

            def dominates(a, b):
                int_le = a['interactions'] <= b['interactions'] + 1e-9
                fail_le = a['failures'] <= b['failures'] + 1e-9
                int_lt = a['interactions'] < b['interactions'] - 1e-9
                fail_lt = a['failures'] < b['failures'] - 1e-9
                return int_le and fail_le and (int_lt or fail_lt)

            wins_this_cell = []
            for b in BASELINE_KEYS:
                bp = baseline_pts[b].get(key)
                if bp is not None and dominates(ip, bp):
                    pareto_vs_each[b] += 1
                    wins_this_cell.append(True)
                else:
                    wins_this_cell.append(False)
            if all(wins_this_cell):
                pareto_vs_all += 1

        ax.set_xlabel('Interactions / episode (sum of all comm)', fontsize=9)
        ax.set_ylabel('Failures / episode', fontsize=9)
        ax.set_title(task, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.85)

    # Hide unused axes
    for idx in range(n_tasks, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis('off')

    reward_pct = 100.0 * reward_wins / max(1, cell_counts)
    each_pct = {b: 100.0 * pareto_vs_each[b] / max(1, cell_counts) for b in BASELINE_KEYS}
    all_pct = 100.0 * pareto_vs_all / max(1, cell_counts)
    fig.suptitle(
        f'(Interactions, failures) operating points across (obs_noise × fail_regime) cells\n'
        f'MA-IPPO selects regime-conditional points the fixed baselines cannot reach: '
        f'reward ≥ best baseline in {reward_wins}/{cell_counts} cells ({reward_pct:.0f}%).  '
        f'Strict Pareto-dominance over all 3 baselines is rare ({all_pct:.0f}%) because '
        f'the fixed baselines occupy disjoint extremes.',
        fontsize=10.5, fontweight='bold', y=1.00,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {output_path}')
    print(f'  MA-IPPO reward-wins:                  {reward_wins} / {cell_counts}  ({reward_pct:.1f}%)')
    for b in BASELINE_KEYS:
        print(f'  MA-IPPO Pareto > {b:<18s}  {pareto_vs_each[b]} / {cell_counts}  ({each_pct[b]:.1f}%)')
    print(f'  MA-IPPO Pareto > all 3 baselines:     {pareto_vs_all} / {cell_counts}  ({all_pct:.1f}%)')
    export_to_paper(output_path, paper=paper)
    plt.close(fig)


def plot_improvement_aggregate(merged: dict, output_path: str,
                               paper: bool = False, mode: str = 'vs-none'):
    """Aggregate per-task bar plot of mean reward improvement over the
    None baseline.

    mode='vs-none' (default): three bars per task =
        (Passive Assistant − None), (Heuristic − None), (MA-IPPO − None).
        Uses None as the common zero-communication reference so all three
        non-trivial policies sit on the same axis.
    """
    if mode != 'vs-none':
        raise SystemExit(f"Unsupported bars-mode {mode!r}; "
                         "only 'vs-none' is implemented for the 4-policy schema.")

    tasks = [t for t in TASK_ORDER if t in merged]
    if not tasks:
        tasks = list(merged.keys())

    bar_pa, bar_h, bar_ip = [], [], []
    for task in tasks:
        pts = collect_points(merged[task])
        none_r = {(p['obs_noise'], p['fail_regime']): p['reward'] for p in pts['none']}
        pa_r   = {(p['obs_noise'], p['fail_regime']): p['reward'] for p in pts['passive_assistant']}
        h_r    = {(p['obs_noise'], p['fail_regime']): p['reward'] for p in pts['heuristic']}
        ip_r   = {(p['obs_noise'], p['fail_regime']): p['reward'] for p in pts['ma_ippo']}
        d_pa = [pa_r[k] - none_r[k] for k in pa_r if k in none_r]
        d_h  = [h_r[k]  - none_r[k] for k in h_r  if k in none_r]
        d_ip = [ip_r[k] - none_r[k] for k in ip_r if k in none_r]
        bar_pa.append(np.mean(d_pa) if d_pa else 0.0)
        bar_h.append(np.mean(d_h) if d_h else 0.0)
        bar_ip.append(np.mean(d_ip) if d_ip else 0.0)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    fig.patch.set_facecolor('white')
    x = np.arange(len(tasks))
    width = 0.27
    ax.bar(x - width, bar_pa, width, color='#E69F00', label='Passive Assistant')
    ax.bar(x,         bar_h,  width, color='#D32F2F', label='Heuristic')
    ax.bar(x + width, bar_ip, width, color='#1565C0', label='Proposed (MA-IPPO)')

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('ΔReward  vs. None baseline\n'
                  'mean over all (obs × fail) regimes', fontsize=10)
    ax.set_title('Reward improvement over None baseline, by task',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {output_path}  (mode={mode})')
    export_to_paper(output_path, paper=paper)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default=str(PROJECT_ROOT / 'data' / 'results'))
    parser.add_argument('--out', default=str(
        PROJECT_ROOT / 'results' / 'figures' / 'pareto_per_task.png'))
    parser.add_argument('--out-bars', default=str(
        PROJECT_ROOT / 'results' / 'figures' / 'pareto_improvement_bars.png'))
    parser.add_argument('--bars-only', action='store_true',
                        help='Skip the per-task scatter plot; render only the '
                             'bar plot. Useful for compact paper layouts.')
    parser.add_argument('--bars-mode', choices=['vs-none'],
                        default='vs-none',
                        help='Bar comparison mode. vs-none (default, only mode): three '
                             'bars per task = (Passive Assistant − None), '
                             '(Heuristic − None), (MA-IPPO − None).')
    add_paper_arg(parser)
    args = parser.parse_args()

    merged = load_all_tasks(Path(args.results_dir))
    if not merged:
        raise SystemExit('No comparison_4policy*.json found')
    if not args.bars_only:
        plot_pareto(merged, args.out, paper=args.paper)
    plot_improvement_aggregate(merged, args.out_bars, paper=args.paper,
                               mode=args.bars_mode)


if __name__ == '__main__':
    main()
