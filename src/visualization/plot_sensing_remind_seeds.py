#!/usr/bin/env python3
"""Seed-aggregated E3-v2 (brittleness × c_remind) figure.

Loads sensing_remind_grid_*.json from multiple seeds and renders
medians + IQR. Same panel structure as plot_sensing_remind_sweep.py
but with multi-seed aggregation.

Usage:
    python plot_sensing_remind_seeds.py \
        --results 'data/results/sensing_remind_grid_make_cereal_cf15_n0.40_seed*.json'
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


def load_seeds(paths):
    grids = []
    meta = None
    for path in paths:
        with open(path) as f:
            d = json.load(f)
        lambda_vals = np.array(d['lambda_vals'])
        cr_vals = np.array(d['c_remind_vals'])
        n_lam, n_cr = len(lambda_vals), len(cr_vals)
        nar = np.full((n_lam, n_cr), np.nan)
        q = np.full((n_lam, n_cr), np.nan)
        rem = np.full((n_lam, n_cr), np.nan)
        con = np.full((n_lam, n_cr), np.nan)
        reward = np.full((n_lam, n_cr), np.nan)
        failures = np.full((n_lam, n_cr), np.nan)
        tracking = np.full((n_lam, n_cr), np.nan)
        for row in d['grid']:
            for cell in row:
                if cell is None:
                    continue
                i, j = cell['i_lambda'], cell['i_c_remind']
                nar[i, j] = cell['mean_narrations']
                q[i, j] = cell['mean_questions']
                rem[i, j] = cell['mean_reminds']
                con[i, j] = cell['mean_confirms']
                reward[i, j] = cell['mean_reward']
                failures[i, j] = cell['mean_failures']
                tracking[i, j] = cell.get('mean_tracking_map_acc', np.nan)
        grids.append({'nar': nar, 'q': q, 'rem': rem, 'con': con,
                      'reward': reward, 'failures': failures,
                      'tracking': tracking, 'seed': d.get('seed', -1)})
        meta = d
    if not grids:
        raise SystemExit('No seed files matched.')
    keys = ['nar', 'q', 'rem', 'con', 'reward', 'failures', 'tracking']
    stacked = {k: np.stack([g[k] for g in grids], axis=0) for k in keys}
    return meta, stacked, [g['seed'] for g in grids]


import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from _paper_export import add_paper_arg, export_to_paper


def plot(paths, output_path=None, paper=False):
    meta, S, seeds = load_seeds(paths)
    lambda_vals = np.array(meta['lambda_vals'])
    cr_vals = np.array(meta['c_remind_vals'])
    n_lam, n_cr = S['nar'].shape[1:]
    n_seeds = S['nar'].shape[0]

    M = {k: np.nanmedian(v, axis=0) for k, v in S.items()}
    p25 = {k: np.nanpercentile(v, 25, axis=0) for k, v in S.items()}
    p75 = {k: np.nanpercentile(v, 75, axis=0) for k, v in S.items()}

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.95],
                          hspace=0.5, wspace=0.4)
    fig.patch.set_facecolor('white')

    x_ticks = np.arange(n_cr)
    y_ticks = np.arange(n_lam)
    x_lbl = [f'{v:.2f}' for v in cr_vals]
    y_lbl = [f'{v:.3f}' for v in lambda_vals]

    def setup(ax, title,
              xlabel='c_remind', ylabel='λ_n  (durable→brittle)'):
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_lbl, fontsize=8, rotation=45)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_lbl, fontsize=8)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')

    # Row 1: medians
    for col, (key, name, cmap) in enumerate([
        ('nar', 'Narrations / ep (median)',  'YlOrRd'),
        ('q',   'Questions / ep (median)',   'YlGn'),
        ('rem', 'Reminds / ep (median)',     'PuRd'),
        ('con', 'Confirms / ep (median)',    'BuPu'),
    ]):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(M[key], origin='lower', aspect='auto', cmap=cmap, vmin=0)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        setup(ax, name)

    # Row 2: reward / failures / tracking / total comm (medians)
    total_int_med = M['nar'] + M['q'] + M['rem'] + M['con']
    for col, (mat, name, cmap, sym) in enumerate([
        (M['reward'], 'Median reward', 'RdBu_r', True),
        (M['failures'], 'Median failures / ep', 'Reds', False),
        (M['tracking'], 'Median tracking acc.', 'viridis', False),
        (total_int_med, 'Total comm. / ep (median)', 'YlGnBu', False),
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

    # Row 3: line plots with IQR shading at fixed cheapest cr (vary λ)
    ax = fig.add_subplot(gs[2, :2])
    j_cheap = 0
    for key, color, label in [
        ('nar', '#E65100', 'Narrations'),
        ('q',   '#FFA000', 'Questions'),
        ('rem', '#2E7D32', 'Reminders'),
        ('con', '#1565C0', 'Confirms'),
    ]:
        ax.plot(lambda_vals, M[key][:, j_cheap], '-o', color=color,
                label=label, linewidth=2)
        ax.fill_between(lambda_vals, p25[key][:, j_cheap],
                        p75[key][:, j_cheap], color=color, alpha=0.18)
    ax.set_xscale('log')
    ax.set_xlabel(f'λ_n   [at c_remind={cr_vals[j_cheap]:.2f}]', fontsize=10)
    ax.set_ylabel('actions / episode (median ± IQR)', fontsize=10)
    ax.set_title('Role-swap at cheap reminder cost', fontsize=11,
                 fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 2:])
    i_brittle = n_lam - 1
    for key, color, label in [
        ('nar', '#E65100', 'Narrations'),
        ('q',   '#FFA000', 'Questions'),
        ('rem', '#2E7D32', 'Reminders'),
        ('con', '#1565C0', 'Confirms'),
    ]:
        ax.plot(cr_vals, M[key][i_brittle, :], '-o', color=color,
                label=label, linewidth=2)
        ax.fill_between(cr_vals, p25[key][i_brittle, :],
                        p75[key][i_brittle, :], color=color, alpha=0.18)
    ax.set_xscale('log')
    ax.set_xlabel(f'c_remind   [at λ_n={lambda_vals[i_brittle]:.3f} (brittle)]',
                  fontsize=10)
    ax.set_ylabel('actions / episode (median ± IQR)', fontsize=10)
    ax.set_title('Reminder activation as cost falls', fontsize=11,
                 fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'Sensing × reminder-cost (multi-seed)   |   '
        f'task: {meta.get("task")}   n_base={meta.get("obs_noise")}   '
        f'seeds={seeds} ({n_seeds} total)',
        fontsize=12, fontweight='bold', y=1.00,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path is None:
        output_path = str(
            PROJECT_ROOT / 'results' / 'figures' /
            f'sensing_remind_sweep_{meta.get("task","")}_seeds_aggregated.png'
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {output_path}')
    export_to_paper(output_path, paper=paper)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', nargs='+', required=True)
    p.add_argument('--output', default=None)
    add_paper_arg(p)
    args = p.parse_args()
    paths = []
    for r in args.results:
        paths.extend(sorted(glob.glob(r)))
    if not paths:
        paths = list(args.results)
    plot(paths, args.output, paper=args.paper)


if __name__ == '__main__':
    main()
