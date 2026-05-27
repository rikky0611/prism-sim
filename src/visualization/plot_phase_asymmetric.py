#!/usr/bin/env python3
"""Cost-asymmetry phase diagram (E1).

Reads grid_asymmetric_*.json (c_nar × c_remind sweep) and produces:
  - 4-class categorical phase diagram
      (Silent / Human-led / Assistant-led / Mixed)
  - Division-of-labor index heatmap  ((nar+q) / total_comm)
  - Per-action heatmaps: narrations, questions, reminds, confirms
  - Antidiagonal slice line plot showing the role-swap transition

Usage:
    python plot_phase_asymmetric.py \
        --results data/results/grid_asymmetric_make_cereal_step_transition_durable_cf15_seed0.json \
        [--output results/figures/phase_asymmetric.png]
        [--threshold 1.0]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).parent.parent.parent

# 4-class categorical palette
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


def load_grid(path):
    with open(path) as f:
        d = json.load(f)
    c_nar = np.array(d['c_nar_vals'])
    c_rem = np.array(d['c_remind_vals'])
    n_rem = len(c_rem)
    n_nar = len(c_nar)

    nar = np.zeros((n_rem, n_nar))
    q = np.zeros((n_rem, n_nar))
    rem = np.zeros((n_rem, n_nar))
    con = np.zeros((n_rem, n_nar))
    reward = np.zeros((n_rem, n_nar))
    failures = np.zeros((n_rem, n_nar))

    for row in d['grid']:
        for cell in row:
            if cell is None:
                continue
            i = cell['i_remind']
            j = cell['i_nar']
            nar[i, j] = cell['mean_narrations']
            q[i, j] = cell['mean_questions']
            rem[i, j] = cell['mean_reminds']
            con[i, j] = cell['mean_confirms']
            reward[i, j] = cell['mean_reward']
            failures[i, j] = cell['mean_failures']

    return d, c_nar, c_rem, nar, q, rem, con, reward, failures


import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from _paper_export import add_paper_arg, export_to_paper


def plot(results_path, output_path=None, threshold=1.0, paper=False,
         no_contours=False):
    d, c_nar, c_rem, nar, q, rem, con, reward, failures = load_grid(results_path)
    task = d.get('task', 'unknown')
    n_rem, n_nar = nar.shape

    # Division-of-labor index
    total_comm = nar + q + rem + con
    dol = np.where(total_comm > 1e-6, (nar + q) / np.maximum(total_comm, 1e-6), np.nan)

    # Categorical phase
    cat_idx = np.zeros((n_rem, n_nar), dtype=int)
    for i in range(n_rem):
        for j in range(n_nar):
            cat_idx[i, j] = CAT_INDEX[classify(nar[i, j], q[i, j], rem[i, j], con[i, j], threshold)]

    cmap_cat = mcolors.ListedColormap(CAT_COLORS)
    norm_cat = mcolors.BoundaryNorm(np.arange(-0.5, len(CATEGORIES)), len(CATEGORIES))

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.7], hspace=0.45, wspace=0.32)
    fig.patch.set_facecolor('white')

    x_ticks = np.arange(n_nar)
    y_ticks = np.arange(n_rem)
    x_lbl = [f'{v:.2f}' if v < 1 else f'{v:.1f}' for v in c_nar]
    y_lbl = [f'{v:.2f}' if v < 1 else f'{v:.1f}' for v in c_rem]

    def setup(ax, title, xlabel='c_nar  (human price)', ylabel='c_remind  (assistant price)'):
        step_x = max(1, n_nar // 6)
        step_y = max(1, n_rem // 6)
        ax.set_xticks(x_ticks[::step_x])
        ax.set_xticklabels([x_lbl[k] for k in x_ticks[::step_x]], fontsize=8, rotation=45)
        ax.set_yticks(y_ticks[::step_y])
        ax.set_yticklabels([y_lbl[k] for k in y_ticks[::step_y]], fontsize=8)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_aspect('auto')

    # Row 1: phase + DoL + reward + failures
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(cat_idx, origin='lower', aspect='auto', cmap=cmap_cat, norm=norm_cat,
              interpolation='nearest')
    if n_rem > 2 and n_nar > 2 and not no_contours:
        try:
            ax.contour(cat_idx, levels=np.arange(0.5, len(CATEGORIES)),
                       colors='white', linewidths=0.7, alpha=0.7)
        except Exception:
            pass
    setup(ax, f'Phase  (thresh={threshold})')
    present = sorted(set(cat_idx.flatten().tolist()))
    patches = [Patch(color=CAT_COLORS[i], label=CAT_LABELS[i]) for i in present]
    ax.legend(handles=patches, loc='upper right', fontsize=8, framealpha=0.85,
              edgecolor='#ccc')

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(dol, origin='lower', aspect='auto', cmap='RdBu_r',
                   vmin=0, vmax=1, interpolation='nearest')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    setup(ax, 'Division-of-labor  (human share)')

    ax = fig.add_subplot(gs[0, 2])
    vmax_r = max(abs(reward.min()), abs(reward.max())) if reward.size else 1.0
    im = ax.imshow(reward, origin='lower', aspect='auto', cmap='RdBu_r',
                   vmin=-vmax_r, vmax=vmax_r)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    setup(ax, 'Mean reward')

    ax = fig.add_subplot(gs[0, 3])
    im = ax.imshow(failures, origin='lower', aspect='auto', cmap='Reds', vmin=0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    setup(ax, 'Failures / episode')

    # Row 2: per-action heatmaps
    for col, (mat, name, cmap) in enumerate([
        (nar, 'Narrations / ep',  'YlOrRd'),
        (q,   'Questions / ep',   'YlGn'),
        (rem, 'Reminds / ep',     'PuRd'),
        (con, 'Confirms / ep',    'BuPu'),
    ]):
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(mat, origin='lower', aspect='auto', cmap=cmap, vmin=0)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if not no_contours:
            try:
                ax.contour(mat, levels=[threshold], colors='black',
                           linewidths=1.0, linestyles='--', alpha=0.5)
            except Exception:
                pass
        setup(ax, name)

    # Row 3: antidiagonal slice — sharpness of role swap
    ax = fig.add_subplot(gs[2, :])
    # Walk along the antidiagonal: c_nar increasing while c_remind decreasing
    diag_len = min(n_rem, n_nar)
    diag_idx = np.linspace(0, diag_len - 1, diag_len).astype(int)
    nar_slice = np.array([nar[n_rem - 1 - k, k] for k in diag_idx])
    q_slice   = np.array([q  [n_rem - 1 - k, k] for k in diag_idx])
    rem_slice = np.array([rem[n_rem - 1 - k, k] for k in diag_idx])
    con_slice = np.array([con[n_rem - 1 - k, k] for k in diag_idx])
    dol_slice = np.array([dol[n_rem - 1 - k, k] for k in diag_idx])
    x = np.arange(diag_len)

    ax.plot(x, nar_slice + q_slice,    '-o', color='#E65100', label='Human (narr+q)', linewidth=2)
    ax.plot(x, rem_slice + con_slice,  '-s', color='#2E7D32', label='Assistant (rem+con)', linewidth=2)
    ax.set_xlabel('antidiagonal step  (left: cheap human, expensive assistant   →   right: expensive human, cheap assistant)',
                  fontsize=9)
    ax.set_ylabel('communicative actions / episode', fontsize=10)
    ax.set_title('Antidiagonal slice — role-swap as cost shifts between agents',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper center', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)

    # Twin axis for DoL
    ax2 = ax.twinx()
    ax2.plot(x, dol_slice, ':', color='#444', linewidth=1.5, label='Human share')
    ax2.set_ylabel('Division-of-labor (human share)', fontsize=9, color='#444')
    ax2.set_ylim(-0.05, 1.05)

    # Suptitle
    fig.suptitle(
        f'Cost-Asymmetry Phase Diagram   |   task: {task}   '
        f'c_fail_scale={d.get("c_fail_scale",15)}   '
        f'decay={d.get("decay_regime","")}   obs={d.get("obs_regime","")}   '
        f'seed={d.get("seed",0)}',
        fontsize=12, fontweight='bold', y=0.995,
    )

    if output_path is None:
        output_path = str(
            PROJECT_ROOT / 'results' / 'figures' /
            f'phase_asymmetric_{task}_seed{d.get("seed",0)}.png'
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {output_path}')
    export_to_paper(output_path, paper=paper)
    plt.close(fig)

    print(f'\nPhase distribution (threshold={threshold}):')
    for i, (name, label, color) in enumerate(CATEGORIES):
        count = int(np.sum(cat_idx == i))
        if count > 0:
            pct = 100 * count / (n_rem * n_nar)
            print(f'  {label:18s}  {count:3d} / {n_rem*n_nar}  ({pct:.1f}%)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--threshold', type=float, default=1.0)
    parser.add_argument('--no-contours', action='store_true',
                        help='Suppress white phase-boundary contours and '
                             'dotted threshold contours (cleaner compact view).')
    add_paper_arg(parser)
    args = parser.parse_args()
    plot(args.results, args.output, args.threshold, paper=args.paper,
         no_contours=args.no_contours)


if __name__ == '__main__':
    main()
