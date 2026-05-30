#!/usr/bin/env python3
"""Sensing-fidelity sweep plot (E3).

Reads sensing_grid_*.json (obs_noise × lambda_noise_recover sweep) and produces:
  - Heatmaps:  narrations, reminds, confirms, tracking accuracy
  - Role-swap line plot at fixed obs_noise: nar/rem/con vs lambda_n
  - Tracking-accuracy contour overlaid on narration heatmap

Tests claim C3: as narration becomes brittle (lambda ↑), the assistant takes over.

Usage:
    python plot_sensing_sweep.py \
        --results data/results/sensing_grid_make_cereal_cf15_cc0.50_seed0.json \
        [--output results/figures/sensing_sweep.png]
        [--noise-slice 0.5]
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
    noise_vals = np.array(d['noise_vals'])
    lambda_vals = np.array(d['lambda_vals'])
    n_lam = len(lambda_vals)
    n_noise = len(noise_vals)

    nar = np.zeros((n_lam, n_noise))
    q = np.zeros((n_lam, n_noise))
    rem = np.zeros((n_lam, n_noise))
    con = np.zeros((n_lam, n_noise))
    reward = np.zeros((n_lam, n_noise))
    failures = np.zeros((n_lam, n_noise))
    tracking = np.full((n_lam, n_noise), np.nan)

    for row in d['grid']:
        for cell in row:
            if cell is None:
                continue
            i = cell['i_lambda']
            j = cell['i_noise']
            nar[i, j] = cell['mean_narrations']
            q[i, j] = cell['mean_questions']
            rem[i, j] = cell['mean_reminds']
            con[i, j] = cell['mean_confirms']
            reward[i, j] = cell['mean_reward']
            failures[i, j] = cell['mean_failures']
            tracking[i, j] = cell.get('mean_tracking_map_acc', np.nan)

    return d, noise_vals, lambda_vals, nar, q, rem, con, reward, failures, tracking


import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from _paper_export import add_paper_arg, export_to_paper, V5_FAILURE_TAG


def plot(results_path, output_path=None, noise_slice=None, paper=False):
    (d, noise_vals, lambda_vals, nar, q, rem, con,
     reward, failures, tracking) = load_grid(results_path)
    task = d.get('task', 'unknown')
    n_lam, n_noise = nar.shape

    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.8], hspace=0.5, wspace=0.32)
    fig.patch.set_facecolor('white')

    x_ticks = np.arange(n_noise)
    y_ticks = np.arange(n_lam)
    x_lbl = [f'{v:.2f}' for v in noise_vals]
    y_lbl = [f'{v:.3f}' for v in lambda_vals]

    def setup(ax, title):
        step_x = max(1, n_noise // 6)
        step_y = max(1, n_lam // 6)
        ax.set_xticks(x_ticks[::step_x])
        ax.set_xticklabels([x_lbl[k] for k in x_ticks[::step_x]], fontsize=8, rotation=45)
        ax.set_yticks(y_ticks[::step_y])
        ax.set_yticklabels([y_lbl[k] for k in y_ticks[::step_y]], fontsize=8)
        ax.set_xlabel('obs_noise  (baseline)', fontsize=10)
        ax.set_ylabel('λ_n  (← durable | brittle →)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_aspect('auto')

    # Row 1: narrations, reminds, confirms, tracking
    for col, (mat, name, cmap) in enumerate([
        (nar,      'Narrations / ep',       'YlOrRd'),
        (rem,      'Reminds / ep',          'PuRd'),
        (con,      'Confirms / ep',         'BuPu'),
        (tracking, 'Tracking MAP accuracy', 'Greens'),
    ]):
        ax = fig.add_subplot(gs[0, col])
        if name == 'Tracking MAP accuracy':
            im = ax.imshow(mat, origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=1)
        else:
            im = ax.imshow(mat, origin='lower', aspect='auto', cmap=cmap, vmin=0)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        setup(ax, name)

    # Row 2: questions, reward, failures, narration with tracking contour
    for col, (mat, name, cmap, kw) in enumerate([
        (q,        'Questions / ep',  'YlGn',   dict(vmin=0)),
        (reward,   'Mean reward',     'RdBu_r', dict()),
        (failures, 'Failures / ep',   'Reds',   dict(vmin=0)),
    ]):
        ax = fig.add_subplot(gs[1, col])
        if 'vmin' not in kw and name == 'Mean reward':
            vmax_r = max(abs(reward.min()), abs(reward.max())) if reward.size else 1.0
            kw['vmin'] = -vmax_r
            kw['vmax'] = vmax_r
        im = ax.imshow(mat, origin='lower', aspect='auto', cmap=cmap, **kw)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        setup(ax, name)

    # Last panel of row 2: narration heatmap with tracking contours
    ax = fig.add_subplot(gs[1, 3])
    im = ax.imshow(nar, origin='lower', aspect='auto', cmap='YlOrRd', vmin=0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    finite_track = np.isfinite(tracking)
    if finite_track.any() and n_lam > 2 and n_noise > 2:
        levels = sorted(set(np.round(np.linspace(
            float(np.nanmin(tracking)), float(np.nanmax(tracking)), 4), 2)))
        try:
            cs = ax.contour(tracking, levels=levels, colors='black',
                            linewidths=1.0, alpha=0.7)
            ax.clabel(cs, fontsize=7, fmt='%.2f')
        except Exception:
            pass
    setup(ax, 'Narrations  (contours: tracking acc)')

    # Row 3: line plot — role swap at fixed obs_noise
    ax = fig.add_subplot(gs[2, :])
    if noise_slice is None:
        # Pick the column closest to median obs_noise
        target = float(np.median(noise_vals))
        col_idx = int(np.argmin(np.abs(noise_vals - target)))
    else:
        col_idx = int(np.argmin(np.abs(noise_vals - float(noise_slice))))
    chosen_noise = noise_vals[col_idx]

    nar_slice = nar[:, col_idx]
    q_slice = q[:, col_idx]
    rem_slice = rem[:, col_idx]
    con_slice = con[:, col_idx]
    track_slice = tracking[:, col_idx]

    ax.plot(lambda_vals, nar_slice, '-o', color='#E65100', label='Narrations', linewidth=2)
    ax.plot(lambda_vals, q_slice,   '-^', color='#FFA000', label='Questions', linewidth=2)
    ax.plot(lambda_vals, rem_slice, '-s', color='#2E7D32', label='Reminds', linewidth=2)
    ax.plot(lambda_vals, con_slice, '-D', color='#1565C0', label='Confirms', linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel('λ_n  (recovery rate; small = durable narration, large = brittle)', fontsize=10)
    ax.set_ylabel('communicative actions / episode', fontsize=10)
    ax.set_title(f'Role-swap slice at obs_noise = {chosen_noise:.2f}', fontsize=11, fontweight='bold')
    ax.legend(loc='center left', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    # Twin axis: tracking accuracy
    if np.any(np.isfinite(track_slice)):
        ax2 = ax.twinx()
        ax2.plot(lambda_vals, track_slice, ':', color='#666', linewidth=2, label='Tracking acc')
        ax2.set_ylabel('tracking MAP accuracy', fontsize=9, color='#666')
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend(loc='center right', fontsize=9)

    fig.suptitle(
        f'Sensing Fidelity Sweep   |   task: {task}   '
        f'c_comm={d.get("c_comm",0.5):.2f}   {V5_FAILURE_TAG}   '
        f'seed={d.get("seed",0)}',
        fontsize=12, fontweight='bold', y=0.995,
    )

    if output_path is None:
        output_path = str(
            PROJECT_ROOT / 'results' / 'figures' /
            f'sensing_sweep_{task}_seed{d.get("seed",0)}.png'
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
    parser.add_argument('--noise-slice', type=float, default=None,
                        help='obs_noise value to slice for the role-swap line plot')
    add_paper_arg(parser)
    args = parser.parse_args()
    plot(args.results, args.output, args.noise_slice, paper=args.paper)


if __name__ == '__main__':
    main()
