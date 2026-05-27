"""
Figures for the PrISM-style semi-Markov belief sweep.

Reads data/results/comparison_v3_semi_markov_all_tasks.json and emits:

1. tracking_heatmaps_per_policy.png
   For each of {passive, heuristic, ma_ippo}: a heatmap of
   mean(tracking_map_acc) over (obs_noise × fail_regime), averaged across tasks.

2. tracking_heatmaps_per_task.png
   One heatmap per task, showing tracking_map_acc for the passive policy
   (policy-independent env-side belief) as a function of obs_noise × fail_regime.

3. tracking_bars_by_noise.png
   Grouped bars: x=obs_noise, bars per policy, height=mean tracking_map_acc
   averaged over fail_regimes and tasks, with std error bars.

4. reward_bars_by_condition.png
   Reward comparison across the 3 policies in each (noise × fail) cell,
   averaged over tasks.

5. narration_heatmap_ma_ippo.png
   MA-IPPO's learned narration frequency per (obs_noise × fail_regime),
   averaged across tasks.

Usage:
    python3 plot_tracking_sweep.py [--results PATH] [--out-dir PATH]
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.parent

POLICIES = ['passive', 'heuristic', 'ma_ippo']
POLICY_LABELS = {'passive': 'Passive',
                 'heuristic': 'Heuristic',
                 'ma_ippo': 'MA-IPPO'}
POLICY_COLORS = {'passive': '#8c8c8c',
                 'heuristic': '#e8a735',
                 'ma_ippo': '#2b7ab8'}
NOISE_ORDER = ['low', 'normal', 'high']
FAIL_ORDER = ['low', 'balanced', 'high']


def _load(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def _collect_matrix(
    results: Dict, metric: str, policy: str,
    tasks: List[str],
) -> np.ndarray:
    """Return array shape (len(NOISE_ORDER), len(FAIL_ORDER), len(tasks))."""
    out = np.full((len(NOISE_ORDER), len(FAIL_ORDER), len(tasks)), np.nan)
    for ti, task in enumerate(tasks):
        task_data = results['conditions'].get(task, {})
        for ni, noise in enumerate(NOISE_ORDER):
            for fi, fail in enumerate(FAIL_ORDER):
                cell = task_data.get(noise, {}).get(fail, {})
                pol = cell.get(policy)
                if pol is None:
                    continue
                v = pol.get(metric)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    out[ni, fi, ti] = v
    return out


def _heatmap(ax, mat: np.ndarray, title: str, vmin=0.0, vmax=1.0, cmap='viridis'):
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
    ax.set_xticks(range(len(FAIL_ORDER)))
    ax.set_xticklabels([f.capitalize() for f in FAIL_ORDER])
    ax.set_yticks(range(len(NOISE_ORDER)))
    ax.set_yticklabels([n.capitalize() for n in NOISE_ORDER])
    ax.set_xlabel('fail_regime')
    ax.set_ylabel('obs_noise')
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                txt = 'n/a'
            else:
                txt = f'{v:.2f}'
            # pick text color for contrast
            color = 'white' if (not np.isnan(v) and v < (vmin + vmax) / 2) else 'black'
            ax.text(j, i, txt, ha='center', va='center', color=color, fontsize=9)
    return im


def fig_tracking_heatmaps_per_policy(results: Dict, tasks: List[str], out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, policy in zip(axes, POLICIES):
        mat = _collect_matrix(results, 'mean_tracking_map_acc', policy, tasks)
        mean_mat = np.nanmean(mat, axis=2)
        im = _heatmap(ax, mean_mat, f'{POLICY_LABELS[policy]}', vmin=0.3, vmax=1.0)
    fig.suptitle('Tracking MAP accuracy — mean across {} tasks'.format(len(tasks)),
                 fontsize=13)
    fig.colorbar(im, ax=axes, label='mean tracking_map_acc', shrink=0.8)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'wrote {out}')


def fig_tracking_heatmaps_per_task(results: Dict, tasks: List[str], out: Path,
                                    policy: str = 'passive'):
    n = len(tasks)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.3 * rows),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    im = None
    for idx, task in enumerate(tasks):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        mat = _collect_matrix(results, 'mean_tracking_map_acc', policy, [task])
        im = _heatmap(ax, mat[..., 0], task, vmin=0.3, vmax=1.0)
    # hide unused cells
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis('off')
    fig.suptitle(f'Tracking MAP accuracy per task — {POLICY_LABELS[policy]}',
                 fontsize=13)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(),
                     label='tracking_map_acc', shrink=0.75)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'wrote {out}')


def fig_tracking_bars_by_noise(results: Dict, tasks: List[str], out: Path):
    """For each noise level, grouped bars across policies (mean over fail_regimes and tasks)."""
    fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    x = np.arange(len(NOISE_ORDER))
    width = 0.25
    for i, policy in enumerate(POLICIES):
        mat = _collect_matrix(results, 'mean_tracking_map_acc', policy, tasks)
        # shape (noise, fail, task) → aggregate over fail & task
        means = np.nanmean(mat, axis=(1, 2))
        # std across tasks × fails (for error bar)
        stds = np.nanstd(mat.reshape(len(NOISE_ORDER), -1), axis=1)
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=3,
               color=POLICY_COLORS[policy], label=POLICY_LABELS[policy])
    ax.set_xticks(x)
    ax.set_xticklabels([n.capitalize() for n in NOISE_ORDER])
    ax.set_xlabel('Observation noise regime')
    ax.set_ylabel('Tracking MAP accuracy')
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color='lightgray', linestyle='--', linewidth=0.7, zorder=0)
    ax.legend(loc='lower left', frameon=False)
    ax.set_title('Tracking accuracy vs observation noise (avg over fail_regimes & tasks)')
    ax.grid(axis='y', linewidth=0.3, alpha=0.5)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'wrote {out}')


def fig_reward_bars_by_condition(results: Dict, tasks: List[str], out: Path):
    """Grouped bars for mean_reward; one panel per obs_noise, x=fail_regime, bars per policy."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True,
                             constrained_layout=True)
    for ax, noise in zip(axes, NOISE_ORDER):
        x = np.arange(len(FAIL_ORDER))
        width = 0.25
        for i, policy in enumerate(POLICIES):
            mat = _collect_matrix(results, 'mean_reward', policy, tasks)
            vals = np.nanmean(mat[NOISE_ORDER.index(noise), :, :], axis=1)
            stds = np.nanstd(mat[NOISE_ORDER.index(noise), :, :], axis=1)
            ax.bar(x + (i - 1) * width, vals, width, yerr=stds, capsize=2,
                   color=POLICY_COLORS[policy], label=POLICY_LABELS[policy])
        ax.set_xticks(x)
        ax.set_xticklabels([f.capitalize() for f in FAIL_ORDER])
        ax.set_xlabel('fail_regime')
        ax.set_title(f'{noise.capitalize()} noise')
        ax.grid(axis='y', linewidth=0.3, alpha=0.5)
    axes[0].set_ylabel('Mean reward (higher is better)')
    axes[-1].legend(loc='lower left', frameon=False, fontsize=9)
    fig.suptitle('Policy reward comparison (avg over {} tasks)'.format(len(tasks)),
                 fontsize=13)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'wrote {out}')


def fig_narration_heatmap(results: Dict, tasks: List[str], out: Path):
    mat = _collect_matrix(results, 'mean_narrations', 'ma_ippo', tasks)
    mean_mat = np.nanmean(mat, axis=2)
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    im = ax.imshow(mean_mat, cmap='magma', aspect='auto')
    ax.set_xticks(range(len(FAIL_ORDER)))
    ax.set_xticklabels([f.capitalize() for f in FAIL_ORDER])
    ax.set_yticks(range(len(NOISE_ORDER)))
    ax.set_yticklabels([n.capitalize() for n in NOISE_ORDER])
    ax.set_xlabel('fail_regime')
    ax.set_ylabel('obs_noise')
    ax.set_title('MA-IPPO narrations / episode (avg across tasks)')
    for i in range(mean_mat.shape[0]):
        for j in range(mean_mat.shape[1]):
            v = mean_mat[i, j]
            color = 'white' if v < mean_mat.max() / 2 else 'black'
            ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                    color=color, fontsize=10)
    fig.colorbar(im, ax=ax, label='narrations / episode')
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'wrote {out}')


def fig_tracking_vs_narration(results: Dict, tasks: List[str], out: Path):
    """Scatter: narration frequency vs tracking accuracy across MA-IPPO conditions."""
    xs, ys, labs = [], [], []
    for task in tasks:
        for noise in NOISE_ORDER:
            for fail in FAIL_ORDER:
                cell = results['conditions'].get(task, {}).get(noise, {}).get(fail, {})
                pol = cell.get('ma_ippo')
                if pol is None:
                    continue
                xs.append(pol['mean_narrations'])
                ys.append(pol['mean_tracking_map_acc'])
                labs.append((task, noise, fail))
    xs = np.array(xs); ys = np.array(ys)
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    ax.scatter(xs, ys, c='#2b7ab8', alpha=0.7, s=35, edgecolor='k', linewidth=0.3)
    ax.set_xlabel('Narrations / episode (MA-IPPO human)')
    ax.set_ylabel('Tracking MAP accuracy')
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color='lightgray', linestyle='--', linewidth=0.7)
    ax.set_title('More narration → higher tracking accuracy (MA-IPPO)')
    ax.grid(linewidth=0.3, alpha=0.5)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f'wrote {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(
        PROJECT_ROOT / 'data' / 'results' /
        'comparison_v3_semi_markov_all_tasks.json'))
    ap.add_argument('--out-dir', default=str(
        PROJECT_ROOT / 'results' / 'figures' / 'tracking_sweep'))
    args = ap.parse_args()

    results = _load(Path(args.results))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Task order: put available tasks in priority order
    preferred = ['make_cereal', 'make_coffee', 'make_tea', 'make_sandwich',
                 'cooking', 'make_stencil', 'latte_making']
    tasks = [t for t in preferred if t in results['conditions']]
    print(f'Tasks in results: {tasks}')

    fig_tracking_heatmaps_per_policy(results, tasks, out / 'tracking_heatmaps_per_policy.png')
    fig_tracking_heatmaps_per_task(results, tasks, out / 'tracking_heatmaps_per_task.png')
    fig_tracking_bars_by_noise(results, tasks, out / 'tracking_bars_by_noise.png')
    fig_reward_bars_by_condition(results, tasks, out / 'reward_bars_by_condition.png')
    fig_narration_heatmap(results, tasks, out / 'narration_heatmap_ma_ippo.png')
    fig_tracking_vs_narration(results, tasks, out / 'tracking_vs_narration.png')


if __name__ == '__main__':
    main()
