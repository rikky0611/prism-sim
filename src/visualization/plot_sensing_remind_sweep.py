#!/usr/bin/env python3
"""E3-v2 plot: brittleness × reminder-cost grid.

Reads sensing_remind_grid_*.json (lambda_n × c_remind sweep at fixed
n_base) and renders:
  - Per-action heatmaps (narrations, questions, reminds, confirms)
  - Reward + failures heatmaps
  - Role-swap line plot at fixed c_remind (lowest): nar / rem vs lambda_n
  - Role-swap line plot at fixed lambda_n (highest, brittle): nar / rem vs c_remind

Tests claim C3 directly: at low c_remind, as narration becomes brittle
(lambda ↑), reminders should take over.

Usage:
    python plot_sensing_remind_sweep.py \
        --results data/results/sensing_remind_grid_make_cereal_cf15_n0.40_seed0.json
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
    lambda_vals = np.array(d['lambda_vals'])
    cr_vals = np.array(d['c_remind_vals'])
    n_lam = len(lambda_vals)
    n_cr = len(cr_vals)

    def make():
        return np.full((n_lam, n_cr), np.nan)

    nar, q, rem, con = make(), make(), make(), make()
    reward, failures, tracking = make(), make(), make()

    for row in d['grid']:
        for cell in row:
            if cell is None:
                continue
            i = cell['i_lambda']
            j = cell['i_c_remind']
            nar[i, j] = cell['mean_narrations']
            q[i, j] = cell['mean_questions']
            rem[i, j] = cell['mean_reminds']
            con[i, j] = cell['mean_confirms']
            reward[i, j] = cell['mean_reward']
            failures[i, j] = cell['mean_failures']
            tracking[i, j] = cell.get('mean_tracking_map_acc', np.nan)

    return d, lambda_vals, cr_vals, nar, q, rem, con, reward, failures, tracking


import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from _paper_export import add_paper_arg, export_to_paper


def plot(results_path, output_path=None, paper=False):
    d, lambda_vals, cr_vals, nar, q, rem, con, reward, failures, tracking = \
        load_grid(results_path)
    task = d.get('task', 'unknown')
    n_lam, n_cr = nar.shape

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.9],
                          hspace=0.5, wspace=0.4)
    fig.patch.set_facecolor('white')

    x_ticks = np.arange(n_cr)
    y_ticks = np.arange(n_lam)
    x_lbl = [f'{v:.2f}' for v in cr_vals]
    y_lbl = [f'{v:.3f}' for v in lambda_vals]

    def setup(ax, title,
              xlabel='c_remind  (assistant price)',
              ylabel='λ_n  (durable→brittle)'):
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_lbl, fontsize=8, rotation=45)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_lbl, fontsize=8)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')

    # Row 1: per-action heatmaps
    for col, (mat, name, cmap) in enumerate([
        (nar, 'Narrations / ep',  'YlOrRd'),
        (q,   'Questions / ep',   'YlGn'),
        (rem, 'Reminds / ep',     'PuRd'),
        (con, 'Confirms / ep',    'BuPu'),
    ]):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(mat, origin='lower', aspect='auto', cmap=cmap, vmin=0)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        setup(ax, name)

    # Row 2: reward, failures, tracking, total interactions
    total_int = nar + q + rem + con
    for col, (mat, name, cmap, sym) in enumerate([
        (reward, 'Mean reward', 'RdBu_r', True),
        (failures, 'Failures / ep', 'Reds', False),
        (tracking, 'Tracking accuracy', 'viridis', False),
        (total_int, 'Total comm. / ep', 'YlGnBu', False),
    ]):
        ax = fig.add_subplot(gs[1, col])
        if sym:
            v = np.nanmax(np.abs(mat)) if mat.size else 1.0
            im = ax.imshow(mat, origin='lower', aspect='auto', cmap=cmap,
                           vmin=-v, vmax=v)
        else:
            im = ax.imshow(mat, origin='lower', aspect='auto', cmap=cmap, vmin=0)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        setup(ax, name)

    # Row 3: role-swap line plots
    # (a) at fixed cheapest c_remind, vary lambda
    ax = fig.add_subplot(gs[2, :2])
    j_cheap = 0  # cheapest c_remind
    ax.plot(lambda_vals, nar[:, j_cheap], '-o', color='#E65100',
            label='Narrations', linewidth=2)
    ax.plot(lambda_vals, q[:, j_cheap],   '-^', color='#FFA000',
            label='Questions', linewidth=2)
    ax.plot(lambda_vals, rem[:, j_cheap], '-s', color='#2E7D32',
            label='Reminders', linewidth=2)
    ax.plot(lambda_vals, con[:, j_cheap], '-d', color='#1565C0',
            label='Confirms', linewidth=2)
    ax.set_xlabel(f'λ_n  (durable → brittle)   '
                  f'[at c_remind={cr_vals[j_cheap]:.2f}]', fontsize=10)
    ax.set_ylabel('actions / episode', fontsize=10)
    ax.set_title('Role-swap at cheap reminder cost: '
                 'as narration becomes brittle, '
                 'does the assistant take over?',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # (b) at fixed brittlest lambda, vary c_remind
    ax = fig.add_subplot(gs[2, 2:])
    i_brittle = n_lam - 1
    ax.plot(cr_vals, nar[i_brittle, :], '-o', color='#E65100',
            label='Narrations', linewidth=2)
    ax.plot(cr_vals, q  [i_brittle, :], '-^', color='#FFA000',
            label='Questions', linewidth=2)
    ax.plot(cr_vals, rem[i_brittle, :], '-s', color='#2E7D32',
            label='Reminders', linewidth=2)
    ax.plot(cr_vals, con[i_brittle, :], '-d', color='#1565C0',
            label='Confirms', linewidth=2)
    ax.set_xlabel(f'c_remind  (cheap → expensive)   '
                  f'[at λ_n={lambda_vals[i_brittle]:.3f} (brittle)]',
                  fontsize=10)
    ax.set_ylabel('actions / episode', fontsize=10)
    ax.set_title('Reminder activation as cost falls: '
                 'in brittle-narration regime, '
                 'do reminders displace narration?',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    fig.suptitle(
        f'Sensing × reminder-cost   |   task: {task}   '
        f'n_base={d.get("obs_noise",0.4)}   '
        f'c_nar={d.get("c_nar",0.5)}   c_fail_scale={d.get("c_fail_scale",15)}   '
        f'seed={d.get("seed",0)}',
        fontsize=12, fontweight='bold', y=1.00,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path is None:
        output_path = str(
            PROJECT_ROOT / 'results' / 'figures' /
            f'sensing_remind_sweep_{task}_seed{d.get("seed",0)}.png'
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {output_path}')
    export_to_paper(output_path, paper=paper)
    plt.close(fig)

    # Console summary: did reminders activate?
    print('\nSummary  (at cheapest c_remind, vary λ_n):')
    for i, lam in enumerate(lambda_vals):
        print(f'  λ={lam:.3f}  nar={nar[i,j_cheap]:5.2f}  '
              f'rem={rem[i,j_cheap]:5.2f}  con={con[i,j_cheap]:5.2f}  '
              f'fail={failures[i,j_cheap]:.2f}  trk={tracking[i,j_cheap]:.3f}')
    print('\nSummary  (at brittlest λ_n, vary c_remind):')
    for j, cr in enumerate(cr_vals):
        print(f'  c_remind={cr:.3f}  nar={nar[i_brittle,j]:5.2f}  '
              f'rem={rem[i_brittle,j]:5.2f}  con={con[i_brittle,j]:5.2f}  '
              f'fail={failures[i_brittle,j]:.2f}  trk={tracking[i_brittle,j]:.3f}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', required=True)
    p.add_argument('--output', default=None)
    add_paper_arg(p)
    args = p.parse_args()
    plot(args.results, args.output, paper=args.paper)


if __name__ == '__main__':
    main()
