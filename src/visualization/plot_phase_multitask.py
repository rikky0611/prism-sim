#!/usr/bin/env python3
"""Multi-task phase diagram side-by-side (E2 cross-task figure).

Reads grid_search_*.json from multiple tasks and renders a horizontal
panel of categorical phase diagrams (one per task) over the same
(c_comm × c_fail) plane. Topology preservation = phase boundaries shift
between tasks but the four-class structure is preserved.

Usage:
    python plot_phase_multitask.py \
        --tasks make_cereal cooking latte_making make_stencil \
        --results-dir data/results
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

# Match plot_phase_asymmetric.py 4-class palette for visual consistency
CATEGORIES = [
    ('silent',   'Silent',         '#1565C0'),
    ('human',    'Human-led',      '#E65100'),
    ('assist',   'Assistant-led',  '#2E7D32'),
    ('mixed',    'Mixed',          '#7B1FA2'),
]
CAT_INDEX = {c[0]: i for i, c in enumerate(CATEGORIES)}
CAT_COLORS = [c[2] for c in CATEGORIES]
CAT_LABELS = [c[1] for c in CATEGORIES]


def classify(narr, q, interact, threshold):
    human_active = (narr + q) >= threshold
    asst_active = interact >= threshold
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
    c_comm = np.array(d['c_comm_vals'])
    c_fail = np.array(d['c_fail_vals'])
    n_fail, n_comm = len(c_fail), len(c_comm)
    nar = np.zeros((n_fail, n_comm))
    q = np.zeros((n_fail, n_comm))
    inter = np.zeros((n_fail, n_comm))
    for row in d['grid']:
        for cell in row:
            if cell is None:
                continue
            i = cell['i_fail']
            j = cell['i_comm']
            nar[i, j] = cell['mean_narrations']
            q[i, j] = cell['mean_questions']
            inter[i, j] = cell.get('mean_interactions',
                                   cell.get('mean_reminds', 0.0))
    return d, c_comm, c_fail, nar, q, inter


def find_grid_file(results_dir: Path, task: str):
    """Find a grid_search_<task>_*.json. Prefer step_transition_durable."""
    candidates = sorted(results_dir.glob(f'grid_search_{task}_*.json'))
    pref = [c for c in candidates if 'step_transition_durable' in c.name]
    if pref:
        return pref[0]
    if candidates:
        return candidates[0]
    return None


import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from _paper_export import add_paper_arg, export_to_paper


def plot_multitask(tasks, results_dir, output_path=None, threshold=1.0, paper=False):
    n = len(tasks)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5.5), squeeze=False)
    fig.patch.set_facecolor('white')
    cmap_cat = mcolors.ListedColormap(CAT_COLORS)
    norm_cat = mcolors.BoundaryNorm(np.arange(-0.5, len(CATEGORIES)),
                                    len(CATEGORIES))

    summaries = []
    for k, task in enumerate(tasks):
        ax = axes[0][k]
        path = find_grid_file(Path(results_dir), task)
        if path is None:
            ax.set_title(f'{task}  [no data]', fontsize=11, color='red')
            ax.axis('off')
            continue

        d, c_comm, c_fail, nar, q, inter = load_grid(path)
        n_fail, n_comm = nar.shape
        cat_idx = np.zeros((n_fail, n_comm), dtype=int)
        for i in range(n_fail):
            for j in range(n_comm):
                cat_idx[i, j] = CAT_INDEX[
                    classify(nar[i, j], q[i, j], inter[i, j], threshold)]

        ax.imshow(cat_idx, origin='lower', aspect='auto', cmap=cmap_cat,
                  norm=norm_cat, interpolation='nearest')
        if n_fail > 2 and n_comm > 2:
            try:
                ax.contour(cat_idx, levels=np.arange(0.5, len(CATEGORIES)),
                           colors='white', linewidths=0.6, alpha=0.6)
            except Exception:
                pass

        x_step = max(1, n_comm // 5)
        y_step = max(1, n_fail // 5)
        x_lbl = [f'{v:.2f}' if v < 1 else f'{v:.1f}' for v in c_comm]
        y_lbl = [f'{v:.1f}' for v in c_fail]
        ax.set_xticks(np.arange(n_comm)[::x_step])
        ax.set_xticklabels([x_lbl[i] for i in np.arange(n_comm)[::x_step]],
                           fontsize=8, rotation=45)
        ax.set_yticks(np.arange(n_fail)[::y_step])
        ax.set_yticklabels([y_lbl[i] for i in np.arange(n_fail)[::y_step]],
                           fontsize=8)
        ax.set_xlabel('c_comm', fontsize=10)
        if k == 0:
            ax.set_ylabel('c_fail_scale', fontsize=10)
        n_steps = d.get('n_steps', d.get('task_n_steps', '?'))
        ax.set_title(task, fontsize=11, fontweight='bold')

        dist = {name: int(np.sum(cat_idx == CAT_INDEX[name]))
                for name, _, _ in CATEGORIES}
        total = n_fail * n_comm
        summaries.append((task, dist, total))

    legend_items = [Patch(color=c[2], label=c[1]) for c in CATEGORIES]
    fig.legend(handles=legend_items, loc='lower center',
               ncol=len(CATEGORIES), fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f'Cross-task phase diagrams over (c_comm × c_fail)   '
        f'[threshold={threshold} action/episode]',
        fontsize=12, fontweight='bold', y=1.00,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])

    if output_path is None:
        output_path = str(
            PROJECT_ROOT / 'results' / 'figures' /
            'phase_diagram_multitask.png'
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {output_path}')
    export_to_paper(output_path, paper=paper)
    plt.close(fig)

    print('\nPhase distribution by task:')
    for task, dist, total in summaries:
        line = f'  {task:18s}'
        for name, label, _ in CATEGORIES:
            pct = 100 * dist[name] / max(1, total)
            line += f'  {label}={dist[name]:3d}({pct:.0f}%)'
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+',
                    default=['make_cereal', 'cooking', 'latte_making',
                             'make_stencil'])
    ap.add_argument('--results-dir', default=str(PROJECT_ROOT / 'data' / 'results'))
    ap.add_argument('--output', default=None)
    ap.add_argument('--threshold', type=float, default=1.0)
    add_paper_arg(ap)
    args = ap.parse_args()
    plot_multitask(args.tasks, args.results_dir, args.output, args.threshold,
                   paper=args.paper)


if __name__ == '__main__':
    main()
