#!/usr/bin/env python3
"""Diagnostic: per-round learning curves across the cells of a cost-asymmetry grid.

Reads grid_asymmetric_<task>_*.json that was produced after run_grid_asymmetric.py
started persisting per-cell `training_log` (a list of one dict per round, as
emitted by train_ma_ippo.train_ippo). Renders a 1x4 panel:

  1. Reward (median + IQR band across cells) vs round
  2. Narrations + Reminds (medians + IQR) vs round
  3. Failures (median + IQR) vs round
  4. (final - initial) reward heatmap per cell, surfacing cells where
     training never improved.

Not a paper figure — output goes to results/figures/, NOT paper_compact/.

Usage:
    python plot_grid_learning_curves.py \\
        --results data/results/grid_asymmetric_make_cereal_step_transition_durable_cf15_seed0.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.parent


def collect_series(grid, key):
    """Return (n_cells, n_rounds) array of `key` per round, NaN where missing."""
    rows = []
    for row in grid:
        for cell in row:
            if cell is None or not cell.get('training_log'):
                continue
            rows.append([r.get(key, np.nan) for r in cell['training_log']])
    if not rows:
        return np.zeros((0, 0))
    max_len = max(len(r) for r in rows)
    arr = np.full((len(rows), max_len), np.nan)
    for i, r in enumerate(rows):
        arr[i, :len(r)] = r
    return arr


def band(ax, arr, color, label, clip_lo=None):
    if arr.size == 0:
        return
    rounds = np.arange(1, arr.shape[1] + 1)
    plot_arr = arr if clip_lo is None else np.where(arr < clip_lo, clip_lo, arr)
    med = np.nanmedian(plot_arr, axis=0)
    p25 = np.nanpercentile(plot_arr, 25, axis=0)
    p75 = np.nanpercentile(plot_arr, 75, axis=0)
    ax.plot(rounds, med, color=color, linewidth=2, label=label)
    ax.fill_between(rounds, p25, p75, color=color, alpha=0.18)


def plot(results_path, output_path=None):
    d = json.load(open(results_path))
    task = d.get('task', 'unknown')
    n_rounds = d.get('n_rounds', '?')
    steps = d.get('steps_per_round', '?')
    seed = d.get('seed', 0)
    grid = d['grid']
    n_remind = len(grid)
    n_nar = len(grid[0]) if grid else 0

    reward = collect_series(grid, 'mean_reward')
    failures = collect_series(grid, 'mean_failures')
    narr = collect_series(grid, 'mean_narrations')
    rem = collect_series(grid, 'mean_reminds')

    if reward.size == 0:
        raise SystemExit(
            f'No training_log found in {results_path}. The grid must be produced '
            'by a run_grid_asymmetric.py that persists per-cell training_log.'
        )

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    fig.patch.set_facecolor('white')

    # Panel 1: reward (median + IQR), with a clipped variant overlay so the
    # axis is not eaten by 1e9-magnitude collapses in any single cell.
    ax = axes[0]
    p1 = np.nanpercentile(reward, 1)
    band(ax, reward, '#1565C0', 'reward (median, IQR)', clip_lo=p1)
    ax.axhline(0, color='black', linewidth=0.6, alpha=0.5)
    ax.set_xlabel('round')
    ax.set_ylabel('mean reward / episode')
    ax.set_title('Joint reward')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    note = f'y clipped at 1st pct ({p1:.1f}) to keep band readable'
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=7,
            color='#555', alpha=0.8)

    # Panel 2: action counts (narrations + reminds)
    ax = axes[1]
    band(ax, narr, '#E65100', 'narrations')
    band(ax, rem, '#2E7D32', 'reminds')
    ax.set_xlabel('round')
    ax.set_ylabel('count / episode')
    ax.set_title('Action counts')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)

    # Panel 3: failures
    ax = axes[2]
    band(ax, failures, '#D32F2F', 'failures')
    ax.set_xlabel('round')
    ax.set_ylabel('failures / episode')
    ax.set_title('Failures')
    ax.grid(True, alpha=0.3)

    # Panel 4: (last - first) reward per cell heatmap
    ax = axes[3]
    delta = np.full((n_remind, n_nar), np.nan)
    cidx = 0
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell is None or not cell.get('training_log'):
                continue
            tl = cell['training_log']
            if len(tl) >= 2:
                delta[i, j] = (tl[-1].get('mean_reward', np.nan)
                               - tl[0].get('mean_reward', np.nan))
    # Symmetric colormap around 0; clip extreme values for legibility.
    vmax = np.nanpercentile(np.abs(delta), 95) if np.isfinite(delta).any() else 1.0
    vmax = max(vmax, 1.0)
    im = ax.imshow(delta, origin='lower', aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, interpolation='nearest')
    c_nar_vals = d.get('c_nar_vals', list(range(n_nar)))
    c_rem_vals = d.get('c_remind_vals', list(range(n_remind)))
    x_step = max(1, n_nar // 4); y_step = max(1, n_remind // 4)
    ax.set_xticks(np.arange(n_nar)[::x_step])
    ax.set_xticklabels([f'{c_nar_vals[i]:.2f}' for i in np.arange(n_nar)[::x_step]],
                       rotation=45, fontsize=8)
    ax.set_yticks(np.arange(n_remind)[::y_step])
    ax.set_yticklabels([f'{c_rem_vals[i]:.2f}' for i in np.arange(n_remind)[::y_step]],
                       fontsize=8)
    ax.set_xlabel('c_nar (human)')
    ax.set_ylabel('c_remind (assistant)')
    ax.set_title(f'Δreward = last − first round (clipped ±{vmax:.0f})')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Failure-mode summary
    n_cells = int(np.sum(~np.isnan(delta)))
    n_non_improving = int(np.sum(delta < 0))
    n_collapse = int(np.sum(delta < -100))
    print(f'\n=== {task} (seed={seed})  '
          f'cells with training_log: {n_cells}/{n_remind*n_nar} ===')
    print(f'  rounds={n_rounds}  steps_per_round={steps}')
    if reward.size > 0:
        first_med = np.nanmedian(reward[:, 0])
        last_med = np.nanmedian(reward[:, -1])
        print(f'  reward median: round 1 = {first_med:+.2f}  →  '
              f'round {reward.shape[1]} = {last_med:+.2f}')
    print(f'  cells with Δreward < 0       : {n_non_improving}/{n_cells}  '
          '(training did not improve)')
    print(f'  cells with Δreward < -100    : {n_collapse}/{n_cells}  '
          '(training collapsed)')

    fig.suptitle(
        f'Learning curves across cost-asymmetry grid    task: {task}    '
        f'rounds={n_rounds}  steps={steps}  seed={seed}  '
        f'cells={n_cells}/{n_remind*n_nar}',
        fontsize=11, fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path is None:
        output_path = str(
            PROJECT_ROOT / 'results' / 'figures' /
            f'learning_curves_{task}_seed{seed}_canonical.png'
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'\nSaved: {output_path}')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True,
                    help='Path to grid_asymmetric_<task>_*.json with per-cell training_log')
    ap.add_argument('--output', default=None)
    args = ap.parse_args()
    plot(args.results, args.output)


if __name__ == '__main__':
    main()
