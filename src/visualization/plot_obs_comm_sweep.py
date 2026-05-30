#!/usr/bin/env python3
"""Observation-fidelity x communication-cost sweep visualization.

For each task, reads
  data/results/sweep2d_{task}_acc{50,75,100}_c{0.5,1.58,5.0}_seed0.json
(3 obs accuracies x 3 symmetric comm costs = 9 cells) and renders, per task,
five panels stacked horizontally:

  [Phase]  [narrations]  [questions]  [reminds]  [confirms]

Each panel is a 3x3 grid with comm cost on x (low->high left->right) and
observation accuracy on y (low->high bottom->top, i.e. origin='lower'),
so the bottom-left corner is "cheap-comm + noisy-obs" and the top-right is
"expensive-comm + perfect-obs". Cell values are annotated.

Single-task or multitask (one row per task) output.

Usage:
    python plot_obs_comm_sweep.py --task make_cereal
    python plot_obs_comm_sweep.py --multitask \
        --tasks make_coffee make_sandwich make_cereal make_tea \
                latte_making cooking make_stencil
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

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from _paper_export import add_paper_arg, export_to_paper, V5_FAILURE_TAG

PROJECT_ROOT = Path(__file__).parent.parent.parent

CATEGORIES = [
    ('silent', 'Silent',        '#1565C0'),
    ('human',  'Human-led',     '#E65100'),
    ('assist', 'Assistant-led', '#2E7D32'),
    ('mixed',  'Mixed',         '#7B1FA2'),
]
CAT_INDEX = {c[0]: i for i, c in enumerate(CATEGORIES)}
CAT_COLORS = [c[2] for c in CATEGORIES]
CAT_LABELS = [c[1] for c in CATEGORIES]

ACCS = [50, 75, 100]        # y axis (origin='lower' -> bottom = acc50, top = acc100)
COSTS = [0.5, 1.58, 5.0]    # x axis (left = c0.5, right = c5.0)

METRICS = [
    ('nar', 'Narrations / ep',  'YlOrRd'),
    ('q',   'Questions / ep',   'YlGn'),
    ('rem', 'Reminds / ep',     'PuRd'),
    ('con', 'Confirms / ep',    'BuPu'),
]


def classify(nar, q, rem, con, threshold=1.0):
    human_active = (nar + q) >= threshold
    asst_active = (rem + con) >= threshold
    if human_active and asst_active:
        return 'mixed'
    if human_active:
        return 'human'
    if asst_active:
        return 'assist'
    return 'silent'


def load_task(task):
    """Load the 3x3 sweep2d grid for a task.

    Returns (mats, missing) where mats has (3, 3) float arrays indexed [i_acc, j_cost]:
    nar, q, rem, con, reward, failures. NaN where the cell file is missing.
    """
    nar = np.full((3, 3), np.nan)
    q   = np.full((3, 3), np.nan)
    rem = np.full((3, 3), np.nan)
    con = np.full((3, 3), np.nan)
    reward = np.full((3, 3), np.nan)
    failures = np.full((3, 3), np.nan)
    missing = []
    for i, acc in enumerate(ACCS):
        for j, cost in enumerate(COSTS):
            f = PROJECT_ROOT / 'data' / 'results' / \
                f'sweep2d_{task}_acc{acc}_c{cost}_seed0.json'
            if not f.exists():
                missing.append(f.name)
                continue
            d = json.load(open(f))
            c = d['grid'][0][0]
            nar[i, j] = c['mean_narrations']
            q  [i, j] = c['mean_questions']
            rem[i, j] = c['mean_reminds']
            con[i, j] = c['mean_confirms']
            reward[i, j]   = c['mean_reward']
            failures[i, j] = c['mean_failures']
    return dict(nar=nar, q=q, rem=rem, con=con, reward=reward, failures=failures), missing


def _draw_phase(ax, mats, threshold, vmaxes=None):
    """Top-left categorical phase panel."""
    cat_idx = np.zeros((3, 3), dtype=int)
    for i in range(3):
        for j in range(3):
            if np.isnan(mats['nar'][i, j]):
                cat_idx[i, j] = -1
                continue
            cat_idx[i, j] = CAT_INDEX[classify(
                mats['nar'][i, j], mats['q'][i, j],
                mats['rem'][i, j], mats['con'][i, j],
                threshold,
            )]
    cmap_cat = mcolors.ListedColormap(CAT_COLORS)
    norm_cat = mcolors.BoundaryNorm(np.arange(-0.5, len(CATEGORIES)), len(CATEGORIES))
    display = np.where(cat_idx < 0, 0, cat_idx).astype(int)
    ax.imshow(display, origin='lower', aspect='auto',
              cmap=cmap_cat, norm=norm_cat, interpolation='nearest')
    # Annotate categorical label
    for i in range(3):
        for j in range(3):
            if cat_idx[i, j] < 0:
                ax.text(j, i, '—', ha='center', va='center', fontsize=9, color='white')
                continue
            ax.text(j, i, CAT_LABELS[cat_idx[i, j]],
                    ha='center', va='center', fontsize=8, color='white',
                    fontweight='bold')
    return cat_idx


def _draw_metric(ax, mat, cmap, vmax=None):
    """Per-action heatmap with numeric annotations."""
    if vmax is None:
        vmax = float(np.nanmax(mat)) if np.isfinite(np.nanmax(mat)) else 1.0
        if vmax <= 0:
            vmax = 1.0
    im = ax.imshow(mat, origin='lower', aspect='auto', cmap=cmap,
                   vmin=0, vmax=vmax, interpolation='nearest')
    for i in range(3):
        for j in range(3):
            v = mat[i, j]
            if not np.isfinite(v):
                ax.text(j, i, '—', ha='center', va='center', fontsize=9, color='black')
                continue
            # Choose text color by contrast
            t = v / vmax if vmax > 0 else 0
            color = 'white' if t > 0.55 else 'black'
            ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')
    return im


def _setup_axes(ax, show_xlabels, show_ylabels, title=None):
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    if show_xlabels:
        ax.set_xticklabels([f'{c:g}' for c in COSTS], fontsize=9)
        ax.set_xlabel('comm cost  (c_nar = c_remind)', fontsize=9)
    else:
        ax.set_xticklabels([])
    if show_ylabels:
        ax.set_yticklabels([f'acc{a}' for a in ACCS], fontsize=9)
        ax.set_ylabel('obs fidelity', fontsize=9)
    else:
        ax.set_yticklabels([])
    if title is not None:
        ax.set_title(title, fontsize=10, fontweight='bold')


def plot_task_row(task, fig, gs_row, show_xlabels=True, row_label=None,
                  shared_vmax=None, threshold=1.0):
    """Draw one task's row of 5 panels (Phase + 4 metrics) into gs_row."""
    mats, missing = load_task(task)
    if missing:
        print(f'  WARN {task}: missing {len(missing)} cells: {missing}')

    # 1) Phase
    ax_p = fig.add_subplot(gs_row[0])
    _draw_phase(ax_p, mats, threshold)
    _setup_axes(ax_p, show_xlabels=show_xlabels,
                show_ylabels=True,
                title='Phase' if row_label is None else None)
    if row_label is not None:
        ax_p.set_ylabel(f'{row_label}\nobs fidelity', fontsize=9, fontweight='bold')

    # 2..5) Per-action heatmaps
    for k, (key, name, cmap) in enumerate(METRICS):
        ax = fig.add_subplot(gs_row[k + 1])
        vmax = shared_vmax[key] if shared_vmax is not None else None
        _draw_metric(ax, mats[key], cmap, vmax=vmax)
        _setup_axes(ax, show_xlabels=show_xlabels,
                    show_ylabels=False,
                    title=name if row_label is None else None)
    return mats


def plot_single(task, output_path=None, paper=False, threshold=1.0):
    fig = plt.figure(figsize=(16, 3.6))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(1, 5, wspace=0.18)
    plot_task_row(task, fig, [gs[0, c] for c in range(5)],
                  show_xlabels=True, threshold=threshold)
    fig.suptitle(
        f'Observation x Communication sweep   |   task: {task}   '
        f'{V5_FAILURE_TAG}   seed=0   (sweep2d, symmetric c_nar=c_remind)',
        fontsize=11, fontweight='bold', y=1.02,
    )

    # Phase legend below
    patches = [Patch(color=c[2], label=c[1]) for c in CATEGORIES]
    fig.legend(handles=patches, loc='lower center', ncol=len(CATEGORIES),
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.06))

    if output_path is None:
        output_path = str(PROJECT_ROOT / 'results' / 'figures' /
                          f'obs_comm_sweep_{task}.png')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {output_path}')
    export_to_paper(output_path, paper=paper)
    plt.close(fig)


def plot_multitask(tasks, output_path=None, paper=False, threshold=1.0):
    n = len(tasks)
    fig = plt.figure(figsize=(16, 2.6 * n + 1.0))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(n + 1, 5, height_ratios=[0.18] + [1] * n,
                          wspace=0.18, hspace=0.30)

    # Column headers
    for k, name in enumerate(['Phase'] + [m[1] for m in METRICS]):
        ax_h = fig.add_subplot(gs[0, k])
        ax_h.axis('off')
        ax_h.text(0.5, 0.2, name, ha='center', va='center',
                  fontsize=12, fontweight='bold', transform=ax_h.transAxes)

    # Compute shared vmax per metric across tasks for fair comparison
    all_mats = {key: [] for key, _, _ in METRICS}
    for t in tasks:
        m, _ = load_task(t)
        for key in all_mats:
            all_mats[key].append(m[key])
    shared_vmax = {}
    for key, arrs in all_mats.items():
        stacked = np.concatenate([a.ravel() for a in arrs])
        vmax = float(np.nanmax(stacked))
        shared_vmax[key] = vmax if (np.isfinite(vmax) and vmax > 0) else 1.0

    # Per-task rows
    for r, t in enumerate(tasks):
        is_last = (r == n - 1)
        plot_task_row(
            t, fig,
            [gs[r + 1, c] for c in range(5)],
            show_xlabels=is_last,
            row_label=t,
            shared_vmax=shared_vmax,
            threshold=threshold,
        )

    fig.suptitle(
        f'Observation-fidelity x Communication-cost sweep   |   sweep2d  '
        f'({V5_FAILURE_TAG}, seed=0, symmetric c_nar=c_remind)',
        fontsize=12, fontweight='bold', y=0.995,
    )
    patches = [Patch(color=c[2], label=c[1]) for c in CATEGORIES]
    fig.legend(handles=patches, loc='lower center', ncol=len(CATEGORIES),
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

    if output_path is None:
        output_path = str(PROJECT_ROOT / 'results' / 'figures' /
                          'obs_comm_sweep_multitask.png')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {output_path}')
    export_to_paper(output_path, paper=paper)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', default=None)
    p.add_argument('--multitask', action='store_true')
    p.add_argument('--tasks', nargs='+', default=[
        'make_coffee', 'make_sandwich', 'make_cereal', 'make_tea',
        'latte_making', 'cooking', 'make_stencil',
    ])
    p.add_argument('--threshold', type=float, default=1.0)
    p.add_argument('--output', default=None)
    add_paper_arg(p)
    args = p.parse_args()

    if args.multitask:
        plot_multitask(args.tasks, output_path=args.output,
                       paper=args.paper, threshold=args.threshold)
    elif args.task:
        plot_single(args.task, output_path=args.output,
                    paper=args.paper, threshold=args.threshold)
    else:
        # default: produce both single-task figures and the multitask
        for t in args.tasks:
            plot_single(t, paper=args.paper, threshold=args.threshold)
        plot_multitask(args.tasks, paper=args.paper, threshold=args.threshold)


if __name__ == '__main__':
    main()
