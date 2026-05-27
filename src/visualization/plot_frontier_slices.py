#!/usr/bin/env python3
"""Frontier slice line plot from grid_search_*.json (E2 supplement).

Reads the c_comm × c_fail grid and renders, at a few representative
c_fail levels, line plots showing how (narrations, questions, reminds)
depend on c_comm.

This makes phase transitions explicit as crossing curves: at low c_comm
the cheap human-led modes dominate, and as c_comm rises a different mode
takes over. Total communicative budget tracks c_fail.

Usage:
    python plot_frontier_slices.py \
        --results data/results/grid_search_make_cereal_step_transition_durable.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_grid(path):
    with open(path) as f:
        d = json.load(f)
    c_comm = np.array(d['c_comm_vals'])
    c_fail = np.array(d['c_fail_vals'])
    n_fail = len(c_fail)
    n_comm = len(c_comm)

    nar = np.zeros((n_fail, n_comm))
    q = np.zeros((n_fail, n_comm))
    inter = np.zeros((n_fail, n_comm))
    reward = np.zeros((n_fail, n_comm))
    failures = np.zeros((n_fail, n_comm))

    for row in d['grid']:
        for cell in row:
            if cell is None:
                continue
            i = cell['i_fail']
            j = cell['i_comm']
            nar[i, j] = cell['mean_narrations']
            q[i, j] = cell['mean_questions']
            inter[i, j] = cell.get('mean_interactions', 0.0)
            reward[i, j] = cell['mean_reward']
            failures[i, j] = cell['mean_failures']

    return d, c_comm, c_fail, nar, q, inter, reward, failures


import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from _paper_export import add_paper_arg, export_to_paper


def plot_slices(results_path, output_path=None, n_slices=4, paper=False):
    d, c_comm, c_fail, nar, q, inter, reward, failures = load_grid(results_path)
    task = d.get('task', 'unknown')
    n_fail = len(c_fail)

    # Pick n_slices evenly spaced indices along c_fail
    if n_fail <= n_slices:
        idxs = list(range(n_fail))
    else:
        idxs = np.linspace(0, n_fail - 1, n_slices).astype(int).tolist()

    fig, axes = plt.subplots(1, len(idxs),
                             figsize=(4.5 * len(idxs), 4.0),
                             sharey=True)
    if len(idxs) == 1:
        axes = [axes]
    fig.patch.set_facecolor('white')

    for ax, idx in zip(axes, idxs):
        cf = c_fail[idx]
        ax.plot(c_comm, nar[idx],   '-o', color='#E65100', label='Narrations', linewidth=2)
        ax.plot(c_comm, q[idx],     '-^', color='#FFA000', label='Questions', linewidth=2)
        ax.plot(c_comm, inter[idx], '-s', color='#2E7D32', label='Reminds (interactions)', linewidth=2)
        ax.set_xscale('log')
        ax.set_xlabel('c_comm', fontsize=10)
        ax.set_title(f'c_fail_scale = {cf:.1f}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        if ax is axes[0]:
            ax.set_ylabel('comm. actions / episode', fontsize=10)
            ax.legend(loc='upper right', fontsize=9)

    fig.suptitle(
        f'Frontier slices at fixed failure cost   |   task: {task}   '
        f'decay={d.get("decay_regime","")}   obs={d.get("obs_regime","")}',
        fontsize=12, fontweight='bold', y=1.02,
    )
    fig.tight_layout()

    if output_path is None:
        output_path = str(
            PROJECT_ROOT / 'results' / 'figures' / f'frontier_slices_{task}.png'
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {output_path}')
    export_to_paper(output_path, paper=paper)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--n-slices', type=int, default=4)
    add_paper_arg(parser)
    args = parser.parse_args()
    plot_slices(args.results, args.output, args.n_slices, paper=args.paper)


if __name__ == '__main__':
    main()
