#!/usr/bin/env python3
"""Cross-task comparison figure for MA-IPPO results.

Usage:
    python plot_cross_task.py --pattern "data/results/ma_*_cheap_narrate_step_transition.json"
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).parent.parent.parent

REGIME_COLORS = {
    'extremely_low':  '#1976D2',
    'balanced':       '#388E3C',
    'extremely_high': '#D32F2F',
}
REGIME_LABELS = {
    'extremely_low':  'Low fail cost',
    'balanced':       'Balanced',
    'extremely_high': 'High fail cost',
}
FAIL_REGIMES = ['extremely_low', 'balanced', 'extremely_high']

METRICS = [
    ('mean_reward',       'Reward',           'Mean episode reward'),
    ('mean_narrations',   'Narration',        'Count / episode'),
    ('mean_interactions', 'Remind + Confirm', 'Count / episode'),
    ('mean_failures',     'Failures',         'Count / episode'),
]


def load_all(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError('No files matched: %s' % pattern)
    results = {}
    for f in files:
        d = json.load(open(f))
        task = d['task']
        if 'regimes' not in d:
            continue
        results[task] = d
    return results


def plot_cross_task(pattern, output_path=None):
    data = load_all(pattern)
    tasks = list(data.keys())
    n_tasks = len(tasks)
    n_regimes = len(FAIL_REGIMES)

    x = np.arange(n_tasks)
    bar_w = 0.25
    offsets = np.linspace(-(n_regimes - 1) / 2, (n_regimes - 1) / 2, n_regimes) * bar_w

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 4.0))

    for ax, (key, title, ylabel) in zip(axes, METRICS):
        for regime, offset in zip(FAIL_REGIMES, offsets):
            vals = []
            for task in tasks:
                e = (data[task].get('regimes', {})
                     .get(regime, {})
                     .get('ma_ippo', {})
                     .get('final_eval', {}))
                vals.append(e.get(key, 0.0))

            ax.bar(x + offset, vals, bar_w * 0.9,
                   color=REGIME_COLORS[regime], alpha=0.85,
                   label=REGIME_LABELS[regime])

        ax.set_xticks(x)
        ax.set_xticklabels([t.replace('_', '\n') for t in tasks], fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.grid(True, axis='y', alpha=0.25, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if key != 'mean_reward':
            ax.set_ylim(bottom=0)

    handles = [mpatches.Patch(color=REGIME_COLORS[r], label=REGIME_LABELS[r])
               for r in FAIL_REGIMES]
    fig.legend(handles=handles, loc='upper center', ncol=n_regimes,
               fontsize=9, bbox_to_anchor=(0.5, 1.05), frameon=False)

    first = next(iter(data.values()))
    comm = first.get('comm_regime', '')
    decay = first.get('decay_regime', '')
    fig.suptitle('Cross-task comparison  |  comm=%s  decay=%s' % (comm, decay),
                 fontsize=12, fontweight='bold', y=1.12)

    fig.subplots_adjust(wspace=0.38)

    if output_path is None:
        output_path = str(PROJECT_ROOT / 'results' / 'figures' /
                          ('cross_task_%s_%s.png' % (comm, decay)))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print('Saved: %s' % output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern', default='data/results/ma_*_cheap_narrate_step_transition.json')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    plot_cross_task(args.pattern, args.output)


if __name__ == '__main__':
    main()
