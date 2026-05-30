#!/usr/bin/env python3
"""E1 baseline overlay: MA-IPPO vs Passive vs Heuristic across the (c_nar, c_remind) grid.

Reads grid_asymmetric_*.json after backfill_grid_asymmetric_baselines.py has
populated `passive_*` and `heuristic_*` per cell, and renders a 3x3 panel:

       column = policy  (Passive | Heuristic | MA-IPPO)
       row    = metric  (reward  | total interactions | failures)

The visual claim is: only MA-IPPO's heatmaps respond to the cost axes;
baselines are nearly flat (Passive trivially, Heuristic because it does not
condition on costs). Adaptation = framework's contribution.

Usage:
    python plot_phase_asymmetric_baselines.py \
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


def load_with_baselines(path):
    with open(path) as f:
        d = json.load(f)
    c_nar = np.array(d['c_nar_vals'])
    c_rem = np.array(d['c_remind_vals'])
    n_rem, n_nar = len(c_rem), len(c_nar)

    def empty():
        return {p: np.full((n_rem, n_nar), np.nan)
                for p in ('passive', 'heuristic', 'ma_ippo')}

    R   = empty()
    INT = empty()  # total interactions per episode
    F   = empty()

    for row in d['grid']:
        for cell in row:
            if cell is None:
                continue
            i, j = cell['i_remind'], cell['i_nar']
            # MA-IPPO
            R['ma_ippo'][i, j]   = cell['mean_reward']
            F['ma_ippo'][i, j]   = cell['mean_failures']
            INT['ma_ippo'][i, j] = (cell['mean_narrations'] + cell['mean_questions']
                                    + cell['mean_reminds'] + cell['mean_confirms'])
            for pol in ('passive', 'heuristic'):
                if cell.get(f'{pol}_mean_reward') is None:
                    continue
                R[pol][i, j] = cell[f'{pol}_mean_reward']
                F[pol][i, j] = cell[f'{pol}_mean_failures']
                INT[pol][i, j] = (cell[f'{pol}_mean_narrations']
                                  + cell[f'{pol}_mean_questions']
                                  + cell[f'{pol}_mean_reminds']
                                  + cell[f'{pol}_mean_confirms'])

    return d, c_nar, c_rem, R, INT, F


import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from _paper_export import add_paper_arg, export_to_paper, V5_FAILURE_TAG


def plot(results_path, output_path=None, paper=False, metrics=None,
         diff_from=None):
    d, c_nar, c_rem, R, INT, F = load_with_baselines(results_path)
    task = d.get('task', 'unknown')
    n_rem, n_nar = R['ma_ippo'].shape

    # When diff_from is set, replace each policy matrix by (policy − ref)
    # and drop the reference column. We also force the colormap to a
    # diverging one centered on zero.
    if diff_from is not None:
        if diff_from not in ('passive', 'heuristic'):
            raise SystemExit(f"Unknown diff-from baseline: {diff_from!r}")
        for D in (R, INT, F):
            ref = D[diff_from].copy()
            for pol in list(D.keys()):
                D[pol] = D[pol] - ref
        POLICIES = [('heuristic', 'Heuristic'),
                    ('ma_ippo',   'Proposed (MA-IPPO)')]
        if diff_from == 'passive':
            POLICIES = [(k, v) for k, v in POLICIES if k != 'passive']
        else:
            POLICIES = [(k, v) for k, v in POLICIES if k != diff_from]
        # All metrics become diff metrics → use diverging colormap & sym scale
        ALL_METRICS = [
            ('reward',       f'Mean reward  −  {diff_from}', R, 'RdBu_r', 'sym'),
            ('interactions', f'Total interactions / ep  −  {diff_from}', INT, 'RdBu_r', 'sym'),
            ('failures',     f'Failures / ep  −  {diff_from}', F, 'RdBu_r', 'sym'),
        ]
    else:
        POLICIES = [('passive', 'Passive'), ('heuristic', 'Heuristic'),
                    ('ma_ippo', 'MA-IPPO')]
        ALL_METRICS = [
            ('reward',       'Mean reward',          R,   'RdBu_r', 'sym'),
            ('interactions', 'Total interactions / ep', INT, 'YlGnBu', 'shared'),
            ('failures',     'Failures / ep',        F,   'Reds',   'shared'),
        ]
    if metrics is None:
        keep_keys = {'reward', 'interactions', 'failures'}
    else:
        keep_keys = {k.strip() for k in metrics}
        unknown = keep_keys - {'reward', 'interactions', 'failures'}
        if unknown:
            raise SystemExit(f"Unknown metric(s): {sorted(unknown)}. "
                             f"Valid: reward, interactions, failures.")
    METRICS = [m[1:] for m in ALL_METRICS if m[0] in keep_keys]
    n_rows = len(METRICS)
    n_cols = len(POLICIES)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.6 * n_cols,
                                      3.7 * n_rows + (0.0 if n_rows > 1 else 0.5)),
                             squeeze=False)
    fig.patch.set_facecolor('white')

    x_ticks = np.arange(n_nar)
    y_ticks = np.arange(n_rem)
    x_lbl = [f'{v:.2f}' if v < 1 else f'{v:.1f}' for v in c_nar]
    y_lbl = [f'{v:.2f}' if v < 1 else f'{v:.1f}' for v in c_rem]
    step_x = max(1, n_nar // 5)
    step_y = max(1, n_rem // 5)

    for i_row, (mname, mat_dict, cmap, scale) in enumerate(METRICS):
        # Compute shared color limits across the row to make rows visually comparable
        all_vals = np.concatenate([
            mat_dict[p].flatten()[~np.isnan(mat_dict[p].flatten())]
            for p, _ in POLICIES
        ])
        if scale == 'sym':
            v = np.nanmax(np.abs(all_vals)) if all_vals.size else 1.0
            vmin, vmax = -v, v
        else:
            vmin, vmax = 0, np.nanmax(all_vals) if all_vals.size else 1.0

        for i_col, (pkey, plabel) in enumerate(POLICIES):
            ax = axes[i_row, i_col]
            mat = mat_dict[pkey]
            im = ax.imshow(mat, origin='lower', aspect='auto', cmap=cmap,
                           vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.set_xticks(x_ticks[::step_x])
            ax.set_xticklabels([x_lbl[k] for k in x_ticks[::step_x]],
                               fontsize=8, rotation=45)
            ax.set_yticks(y_ticks[::step_y])
            ax.set_yticklabels([y_lbl[k] for k in y_ticks[::step_y]], fontsize=8)
            if i_row == n_rows - 1:
                ax.set_xlabel('c_nar  (human price)', fontsize=9)
            if i_col == 0:
                ax.set_ylabel(f'{mname}\n\nc_remind  (assistant price)', fontsize=9)
            if i_row == 0:
                ax.set_title(plabel, fontsize=12, fontweight='bold')

            # Per-axis colorbar on rightmost column
            if i_col == n_cols - 1:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Text overlay: report range to make the "flat vs varying" contrast explicit
            if not np.all(np.isnan(mat)):
                rng = np.nanmax(mat) - np.nanmin(mat)
                ax.text(0.02, 0.98, f'range={rng:.2f}',
                        transform=ax.transAxes, fontsize=8,
                        ha='left', va='top', color='black',
                        bbox=dict(facecolor='white', alpha=0.7, pad=1.5,
                                  edgecolor='none'))

    if diff_from is not None:
        subtitle = (f'Per-cell difference (policy − {diff_from}); red = '
                    f'policy beats {diff_from}, blue = worse.')
        title_prefix = f'Reward improvement over {diff_from} baseline'
    elif n_rows == 1:
        subtitle = (f'Only MA-IPPO\'s reward heatmap responds to the cost axes; '
                    f'baselines are nearly flat (range labels shown).')
        title_prefix = 'Baseline overlay'
    else:
        subtitle = (f'Only MA-IPPO heatmaps respond to the cost axes; baselines '
                    f'are nearly flat in interactions and failures (range '
                    f'labels shown).')
        title_prefix = 'Baseline overlay'
    fig.suptitle(
        f'{title_prefix}   |   task: {task}   '
        f'{V5_FAILURE_TAG}   '
        f'seed={d.get("seed",0)}\n{subtitle}',
        fontsize=11, fontweight='bold', y=1.00,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path is None:
        output_path = str(
            PROJECT_ROOT / 'results' / 'figures' /
            f'phase_asymmetric_baselines_{task}_seed{d.get("seed",0)}.png'
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {output_path}')
    export_to_paper(output_path, paper=paper)
    plt.close(fig)

    # Summary
    print('\nReward range per policy:')
    for pkey, plabel in POLICIES:
        mat = R[pkey]
        if not np.all(np.isnan(mat)):
            print(f'  {plabel:10s}  min={np.nanmin(mat):7.2f}  '
                  f'max={np.nanmax(mat):7.2f}  range={np.nanmax(mat)-np.nanmin(mat):.2f}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', required=True)
    p.add_argument('--output', default=None)
    p.add_argument('--metrics', default=None,
                   help='Comma-separated subset of {reward, interactions, '
                        'failures}. Default: all three.')
    p.add_argument('--diff-from', choices=['passive', 'heuristic'],
                   default=None,
                   help='Render per-cell differences (policy − baseline) '
                        'instead of raw values; drops the baseline column.')
    add_paper_arg(p)
    args = p.parse_args()
    metrics = args.metrics.split(',') if args.metrics else None
    plot(args.results, args.output, paper=args.paper, metrics=metrics,
         diff_from=args.diff_from)


if __name__ == '__main__':
    main()
