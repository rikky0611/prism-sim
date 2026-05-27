#!/usr/bin/env python3
"""Multi-task cost-asymmetry phase diagram side-by-side.

Reads grid_asymmetric_<task>_*.json from multiple tasks and renders a
horizontal panel of categorical phase diagrams (one per task) over the
same (c_nar x c_remind) plane. Demonstrates that the four-class phase
structure (Silent / Human-led / Assistant-led / Mixed) replicates across
procedural tasks of varying length and criticality density, while the
location and area of the phases shift with task structure.

Usage:
    python plot_phase_asymmetric_multitask.py \
        --tasks make_coffee make_sandwich make_cereal make_tea \
                latte_making cooking make_stencil \
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

CATEGORIES = [
    ('silent', 'Silent',        '#1565C0'),
    ('human',  'Human-led',     '#E65100'),
    ('assist', 'Assistant-led', '#2E7D32'),
    ('mixed',  'Mixed',         '#7B1FA2'),
]
CAT_INDEX = {c[0]: i for i, c in enumerate(CATEGORIES)}
CAT_COLORS = [c[2] for c in CATEGORIES]


def classify(narr, q, rem, con, threshold):
    human_active = (narr + q) >= threshold
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
    c_remind = np.array(d['c_remind_vals'])
    n_remind, n_nar = len(c_remind), len(c_nar)
    nar = np.zeros((n_remind, n_nar))
    q = np.zeros((n_remind, n_nar))
    rem = np.zeros((n_remind, n_nar))
    con = np.zeros((n_remind, n_nar))
    for row in d['grid']:
        for cell in row:
            if cell is None:
                continue
            fe = cell.get('final_eval') or cell
            i = cell['i_remind']
            j = cell['i_nar']
            nar[i, j] = fe.get('mean_narrations', 0.0)
            q[i, j] = fe.get('mean_questions', 0.0)
            rem[i, j] = fe.get('mean_reminds', 0.0)
            con[i, j] = fe.get('mean_confirms', 0.0)
    return d, c_nar, c_remind, nar, q, rem, con


def find_grid_file(results_dir: Path, task: str):
    """Find grid_asymmetric_<task>_step_transition_durable_cf*_seed0.json."""
    candidates = sorted(
        results_dir.glob(f'grid_asymmetric_{task}_step_transition_durable_*seed0.json')
    )
    # Exclude archival/backup variants (.lf010, .tier1_pre_floor, etc.)
    candidates = [c for c in candidates if c.name.count('.') == 1]
    return candidates[0] if candidates else None


import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from _paper_export import add_paper_arg, export_to_paper


def plot_multitask(tasks, results_dir, output_path=None, threshold=1.0, paper=False):
    n = len(tasks)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 4.0), squeeze=False)
    fig.patch.set_facecolor('white')
    cmap_cat = mcolors.ListedColormap(CAT_COLORS)
    norm_cat = mcolors.BoundaryNorm(np.arange(-0.5, len(CATEGORIES)),
                                    len(CATEGORIES))

    summaries = []
    for k, task in enumerate(tasks):
        ax = axes[0][k]
        path = find_grid_file(Path(results_dir), task)
        if path is None:
            ax.set_title(f'{task}  [no data]', fontsize=10, color='red')
            ax.axis('off')
            continue

        d, c_nar, c_remind, nar, q, rem, con = load_grid(path)
        n_remind, n_nar = nar.shape
        cat_idx = np.zeros((n_remind, n_nar), dtype=int)
        for i in range(n_remind):
            for j in range(n_nar):
                cat_idx[i, j] = CAT_INDEX[
                    classify(nar[i, j], q[i, j], rem[i, j], con[i, j], threshold)]

        ax.imshow(cat_idx, origin='lower', aspect='auto', cmap=cmap_cat,
                  norm=norm_cat, interpolation='nearest')
        if n_remind > 2 and n_nar > 2:
            try:
                ax.contour(cat_idx, levels=np.arange(0.5, len(CATEGORIES)),
                           colors='white', linewidths=0.6, alpha=0.6)
            except Exception:
                pass

        x_step = max(1, n_nar // 4)
        y_step = max(1, n_remind // 4)
        x_lbl = [f'{v:.1f}' for v in c_nar]
        y_lbl = [f'{v:.1f}' for v in c_remind]
        ax.set_xticks(np.arange(n_nar)[::x_step])
        ax.set_xticklabels([x_lbl[i] for i in np.arange(n_nar)[::x_step]],
                           fontsize=8, rotation=45)
        ax.set_yticks(np.arange(n_remind)[::y_step])
        ax.set_yticklabels([y_lbl[i] for i in np.arange(n_remind)[::y_step]],
                           fontsize=8)
        ax.set_xlabel(r'$c_{\mathrm{nar}}$ (human)', fontsize=9)
        if k == 0:
            ax.set_ylabel(r'$c_{\mathrm{remind}}$ (assistant)', fontsize=9)

        n_steps = d.get('n_steps', '?')
        n_crit = d.get('n_critical', '?')
        ax.set_title(f'{task}', fontsize=10, fontweight='bold')

        dist = {name: int(np.sum(cat_idx == CAT_INDEX[name]))
                for name, _, _ in CATEGORIES}
        total = n_remind * n_nar
        sil_pct = 100 * dist['silent'] / max(1, total)
        ax.text(0.5, -0.30, f'Silent {sil_pct:.0f}%',
                transform=ax.transAxes, ha='center', fontsize=8, color='#555')
        summaries.append((task, dist, total))

    legend_items = [Patch(color=c[2], label=c[1]) for c in CATEGORIES]
    fig.legend(handles=legend_items, loc='lower center',
               ncol=len(CATEGORIES), fontsize=10,
               bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(
        r'Cross-task cost-asymmetry phase diagrams over '
        r'($c_{\mathrm{nar}} \times c_{\mathrm{remind}}$)   '
        f'[threshold={threshold} action/episode]',
        fontsize=11, fontweight='bold', y=1.02,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])

    if output_path is None:
        output_path = str(
            PROJECT_ROOT / 'results' / 'figures' /
            'phase_asymmetric_multitask.png'
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {output_path}')
    export_to_paper(output_path, paper=paper)
    plt.close(fig)

    print('\nPhase distribution by task:')
    for task, dist, total in summaries:
        line = f'  {task:16s}'
        for name, label, _ in CATEGORIES:
            pct = 100 * dist[name] / max(1, total)
            line += f'  {label}={dist[name]:3d}({pct:3.0f}%)'
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+',
                    default=['make_coffee', 'make_sandwich', 'make_cereal',
                             'make_tea', 'latte_making', 'cooking',
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
