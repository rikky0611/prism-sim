#!/usr/bin/env python3
"""Seed-aggregated cost-asymmetry phase diagram (E1, multi-seed).

Loads grid_asymmetric_*.json from multiple seeds and renders the
phase diagram and division-of-labor index using per-cell medians,
plus an IQR-uncertainty panel that shows where seeds disagree.

Usage:
    python plot_phase_asymmetric_seeds.py \
        --results data/results/grid_asymmetric_make_cereal_*_cf15_seed*.json \
        [--threshold 1.0]
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).parent.parent.parent

CATEGORIES = [
    ('silent',   'Silent',         '#1565C0'),
    ('human',    'Human-led',      '#E65100'),
    ('assist',   'Assistant-led',  '#2E7D32'),
    ('mixed',    'Mixed',          '#7B1FA2'),
]
CAT_INDEX = {c[0]: i for i, c in enumerate(CATEGORIES)}
CAT_COLORS = [c[2] for c in CATEGORIES]
CAT_LABELS = [c[1] for c in CATEGORIES]


def classify(nar, q, rem, con, threshold):
    human_active = (nar + q) >= threshold
    asst_active = (rem + con) >= threshold
    if human_active and asst_active:
        return 'mixed'
    if human_active:
        return 'human'
    if asst_active:
        return 'assist'
    return 'silent'


def load_seeds(paths):
    """Stack seeds along axis 0; return shape (S, n_rem, n_nar)."""
    grids = []
    meta = None
    for path in paths:
        with open(path) as f:
            d = json.load(f)
        c_nar = np.array(d['c_nar_vals'])
        c_rem = np.array(d['c_remind_vals'])
        n_rem, n_nar = len(c_rem), len(c_nar)
        nar = np.full((n_rem, n_nar), np.nan)
        q = np.full((n_rem, n_nar), np.nan)
        rem = np.full((n_rem, n_nar), np.nan)
        con = np.full((n_rem, n_nar), np.nan)
        reward = np.full((n_rem, n_nar), np.nan)
        failures = np.full((n_rem, n_nar), np.nan)
        for row in d['grid']:
            for cell in row:
                if cell is None:
                    continue
                i, j = cell['i_remind'], cell['i_nar']
                nar[i, j] = cell['mean_narrations']
                q[i, j] = cell['mean_questions']
                rem[i, j] = cell['mean_reminds']
                con[i, j] = cell['mean_confirms']
                reward[i, j] = cell['mean_reward']
                failures[i, j] = cell['mean_failures']
        grids.append({'nar': nar, 'q': q, 'rem': rem, 'con': con,
                      'reward': reward, 'failures': failures,
                      'seed': d.get('seed', -1)})
        meta = d
    if not grids:
        raise SystemExit('No seed files matched.')
    keys = ['nar', 'q', 'rem', 'con', 'reward', 'failures']
    stacked = {k: np.stack([g[k] for g in grids], axis=0) for k in keys}
    return meta, stacked, [g['seed'] for g in grids]


import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from _paper_export import add_paper_arg, export_to_paper


def plot(paths, output_path=None, threshold=1.0, paper=False):
    meta, S, seeds = load_seeds(paths)
    c_nar = np.array(meta['c_nar_vals'])
    c_rem = np.array(meta['c_remind_vals'])
    n_rem, n_nar = S['nar'].shape[1:]
    n_seeds = S['nar'].shape[0]

    median = {k: np.nanmedian(v, axis=0) for k, v in S.items()}
    p25 = {k: np.nanpercentile(v, 25, axis=0) for k, v in S.items()}
    p75 = {k: np.nanpercentile(v, 75, axis=0) for k, v in S.items()}
    iqr = {k: p75[k] - p25[k] for k in S}

    total_comm = median['nar'] + median['q'] + median['rem'] + median['con']
    dol = np.where(total_comm > 1e-6,
                   (median['nar'] + median['q']) / np.maximum(total_comm, 1e-6),
                   np.nan)

    cat_idx = np.zeros((n_rem, n_nar), dtype=int)
    for i in range(n_rem):
        for j in range(n_nar):
            cat_idx[i, j] = CAT_INDEX[classify(
                median['nar'][i, j], median['q'][i, j],
                median['rem'][i, j], median['con'][i, j], threshold)]

    cmap_cat = mcolors.ListedColormap(CAT_COLORS)
    norm_cat = mcolors.BoundaryNorm(np.arange(-0.5, len(CATEGORIES)),
                                    len(CATEGORIES))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.patch.set_facecolor('white')

    x_ticks = np.arange(n_nar)
    y_ticks = np.arange(n_rem)
    x_lbl = [f'{v:.2f}' if v < 1 else f'{v:.1f}' for v in c_nar]
    y_lbl = [f'{v:.2f}' if v < 1 else f'{v:.1f}' for v in c_rem]

    def setup(ax, title):
        step_x = max(1, n_nar // 6)
        step_y = max(1, n_rem // 6)
        ax.set_xticks(x_ticks[::step_x])
        ax.set_xticklabels([x_lbl[k] for k in x_ticks[::step_x]],
                           fontsize=8, rotation=45)
        ax.set_yticks(y_ticks[::step_y])
        ax.set_yticklabels([y_lbl[k] for k in y_ticks[::step_y]], fontsize=8)
        ax.set_xlabel('c_nar', fontsize=9)
        ax.set_ylabel('c_remind', fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')

    # Row 1: phase + DoL median + DoL IQR
    ax = axes[0, 0]
    ax.imshow(cat_idx, origin='lower', aspect='auto', cmap=cmap_cat,
              norm=norm_cat, interpolation='nearest')
    setup(ax, f'Phase (median, threshold={threshold})')
    legend_items = [Patch(color=c[2], label=c[1]) for c in CATEGORIES]
    ax.legend(handles=legend_items, loc='upper right', fontsize=7,
              framealpha=0.85)

    ax = axes[0, 1]
    im = ax.imshow(dol, origin='lower', aspect='auto', cmap='RdBu_r',
                   vmin=0, vmax=1, interpolation='nearest')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    setup(ax, 'Median division-of-labor (human share)')

    # DoL IQR — compute per-seed DoL then aggregate
    dol_per_seed = []
    for s in range(n_seeds):
        tc = S['nar'][s] + S['q'][s] + S['rem'][s] + S['con'][s]
        d = np.where(tc > 1e-6,
                     (S['nar'][s] + S['q'][s]) / np.maximum(tc, 1e-6),
                     np.nan)
        dol_per_seed.append(d)
    dol_stack = np.stack(dol_per_seed, axis=0)
    dol_iqr = np.nanpercentile(dol_stack, 75, axis=0) - \
              np.nanpercentile(dol_stack, 25, axis=0)

    ax = axes[0, 2]
    im = ax.imshow(dol_iqr, origin='lower', aspect='auto', cmap='Reds',
                   vmin=0, vmax=1, interpolation='nearest')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    setup(ax, 'DoL IQR across seeds  (high = unstable cell)')

    # Row 2: median reward + median failures + reward IQR
    ax = axes[1, 0]
    v = max(abs(np.nanmin(median['reward'])), abs(np.nanmax(median['reward'])))
    im = ax.imshow(median['reward'], origin='lower', aspect='auto',
                   cmap='RdBu_r', vmin=-v, vmax=v)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    setup(ax, 'Median reward')

    ax = axes[1, 1]
    im = ax.imshow(median['failures'], origin='lower', aspect='auto',
                   cmap='Reds', vmin=0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    setup(ax, 'Median failures / episode')

    ax = axes[1, 2]
    im = ax.imshow(iqr['reward'], origin='lower', aspect='auto',
                   cmap='Greys', vmin=0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    setup(ax, 'Reward IQR across seeds')

    fig.suptitle(
        f'Cost-asymmetry phase diagram (multi-seed)   |   '
        f'task: {meta.get("task")}   c_fail_scale={meta.get("c_fail_scale")}   '
        f'seeds={seeds} ({n_seeds} total)',
        fontsize=12, fontweight='bold', y=1.00,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path is None:
        output_path = str(
            PROJECT_ROOT / 'results' / 'figures' /
            f'phase_asymmetric_{meta.get("task","")}_seeds_aggregated.png'
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {output_path}')
    export_to_paper(output_path, paper=paper)
    plt.close(fig)

    print(f'\nAggregated phase distribution (median, threshold={threshold}, '
          f'n_seeds={n_seeds}):')
    for i, (name, label, _) in enumerate(CATEGORIES):
        cnt = int(np.sum(cat_idx == i))
        if cnt > 0:
            pct = 100 * cnt / (n_rem * n_nar)
            print(f'  {label:18s}  {cnt:3d} / {n_rem*n_nar}  ({pct:.1f}%)')

    print(f'\nMean DoL IQR: {np.nanmean(dol_iqr):.3f}  '
          f'(0 = perfect agreement, 1 = full disagreement)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', nargs='+', required=True,
                   help='List or glob of seed JSONs')
    p.add_argument('--output', default=None)
    p.add_argument('--threshold', type=float, default=1.0)
    add_paper_arg(p)
    args = p.parse_args()

    paths = []
    for r in args.results:
        paths.extend(sorted(glob.glob(r)))
    if not paths:
        paths = list(args.results)
    plot(paths, args.output, args.threshold, paper=args.paper)


if __name__ == '__main__':
    main()
