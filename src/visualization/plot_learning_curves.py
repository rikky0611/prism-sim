#!/usr/bin/env python3
"""Plot learning curves from MA-IPPO training logs.

Reads JSON result files and generates a single-row multi-panel figure
showing reward, communication metrics, and failures across training rounds.

Usage:
    python plot_learning_curves.py --results <path_to_json> [--output <path.png>]
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PROJECT_ROOT = Path(__file__).parent.parent.parent

# ---------- Style ----------
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

REGIME_STYLES = {
    'extremely_low': {'color': '#1976D2', 'marker': 's', 'label': 'Low fail cost'},
    'balanced':      {'color': '#388E3C', 'marker': '^', 'label': 'Balanced'},
    'extremely_high': {'color': '#D32F2F', 'marker': 'o', 'label': 'High fail cost'},
}


def load_training_logs(json_path: str) -> Dict[str, Any]:
    with open(json_path) as f:
        return json.load(f)


def extract_series(log: List[Dict], key: str) -> tuple:
    """Extract (rounds, values) from training log."""
    rounds = [e['round'] for e in log]
    values = [e.get(key, 0.0) for e in log]
    return rounds, values


def plot_learning_curves(data: Dict, output_path: str = None):
    """Create a single-row figure with learning curve panels."""
    regimes = data.get('regimes', {})
    task = data.get('task', 'unknown')
    comm_regime = data.get('comm_regime', '')
    decay_regime = data.get('decay_regime', '')
    overrides = data.get('param_overrides', {})

    # ---------- Panel definitions ----------
    # Each panel: (title, series_list)
    # series_list: [(key, style_suffix, sublabel), ...]
    panels = [
        ('Reward', [('mean_reward', '', None)], 'Mean episode reward'),
        ('Narration', [('mean_narrations', '', None)], 'Count / episode'),
        ('Question', [('mean_questions', '', None)], 'Count / episode'),
        ('Remind + Confirm', [('mean_interactions', '', None)], 'Count / episode'),
        ('Failures', [('mean_failures', '', None)], 'Count / episode'),
    ]

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.6 * n_panels, 3.2))

    for ax, (title, series_list, ylabel) in zip(axes, panels):
        for regime_name, regime_data in regimes.items():
            log = regime_data.get('ma_ippo', {}).get('training_log', [])
            if not log:
                continue

            style = REGIME_STYLES.get(regime_name, {})
            color = style.get('color', '#999')
            marker = style.get('marker', 'o')
            base_label = style.get('label', regime_name)

            for key, suffix, sublabel in series_list:
                rounds, values = extract_series(log, key)
                lbl = base_label + ((' ' + sublabel) if sublabel else '')

                ax.plot(rounds, values, linestyle='-', marker=marker,
                        color=color, label=lbl,
                        markersize=4, linewidth=1.8, alpha=0.85,
                        markeredgecolor='white', markeredgewidth=0.5)

                # Reward std band
                if key == 'mean_reward':
                    stds = [e.get('std_reward', 0) for e in log]
                    v, s = np.array(values), np.array(stds)
                    ax.fill_between(rounds, v - s, v + s,
                                    color=color, alpha=0.10)

        ax.set_xlabel('Training round')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=7))
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Y-axis: 0-based for count panels; handle all-zero series
        if 'Count' in ylabel:
            ymin, ymax = ax.get_ylim()
            if ymax < 0.1:  # all-zero series
                ax.set_ylim(-0.05, 1.0)
            else:
                ax.set_ylim(bottom=0)

    # Shared legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center',
                   ncol=len(regimes), fontsize=9,
                   bbox_to_anchor=(0.5, 1.06), frameon=False,
                   columnspacing=2.0, handletextpad=0.5)

    # Title
    override_str = '  '.join('%s=%s' % (k, v) for k, v in overrides.items())
    fig.suptitle('%s  |  comm=%s  decay=%s\n%s' % (
                 task.replace('_', ' '), comm_regime, decay_regime,
                 override_str),
                 fontsize=13, fontweight='bold', y=1.14)

    fig.subplots_adjust(wspace=0.35)

    if output_path is None:
        output_path = str(PROJECT_ROOT / 'results' / 'figures' /
                          ('learning_curves_%s_%s_%s.png' % (task, comm_regime, decay_regime)))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print('Saved: %s' % output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot MA-IPPO learning curves')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to JSON results file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output PNG path (auto-generated if omitted)')
    args = parser.parse_args()

    data = load_training_logs(args.results)
    plot_learning_curves(data, args.output)


if __name__ == '__main__':
    main()
