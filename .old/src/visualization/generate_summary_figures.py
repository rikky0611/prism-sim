"""
Summary Figures Generator

Generates learning curve plots for make_cereal and latte_making experiments.

For each task, outputs a 1×3 subplot figure (one column per cost regime) showing:
  - MA-IPPO mean_reward per training round (blue line + shaded std band)
  - Best baseline reward (red dashed horizontal line)

Outputs:
  results/figures/summary_learning_curves_make_cereal.png
  results/figures/summary_learning_curves_latte_making.png

Usage:
  python3 src/visualization/generate_summary_figures.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'results'
FIGURES_DIR = PROJECT_ROOT / 'results' / 'figures'

REGIME_ORDER = ['extremely_low', 'balanced', 'extremely_high']
REGIME_LABELS = {
    'extremely_low':  'Extremely Low\n(c_fail × 2)',
    'balanced':       'Balanced\n(c_fail × 15)',
    'extremely_high': 'Extremely High\n(c_fail × 50)',
}
REGIME_COLORS = {
    'extremely_low':  '#4a90d9',
    'balanced':       '#e67e22',
    'extremely_high': '#c0392b',
}

TASK_TITLES = {
    'make_cereal':  'make_cereal  (8 steps, 2 critical)',
    'latte_making': 'latte_making  (20 steps, 2 critical)',
}


def load_results(task: str) -> Dict[str, Any]:
    path = DATA_DIR / f'ma_experiments_{task}_all.json'
    with open(path) as f:
        return json.load(f)


def best_baseline_reward(baselines: Dict[str, Any]) -> float:
    return max(v['mean_reward'] for v in baselines.values())


def plot_task_learning_curves(task: str) -> None:
    data = load_results(task)
    regimes = data['regimes']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    fig.suptitle(TASK_TITLES[task], fontsize=14, fontweight='bold', y=1.01)

    for ax, regime in zip(axes, REGIME_ORDER):
        if regime not in regimes:
            ax.set_visible(False)
            continue

        rdata = regimes[regime]
        log = rdata['ma_ippo']['training_log']
        baselines = rdata['baselines']

        rounds = np.array([entry['round'] for entry in log])
        rewards = np.array([entry['mean_reward'] for entry in log])
        stds = np.array([entry.get('std_reward', 0.0) for entry in log])

        best_bl = best_baseline_reward(baselines)
        color = REGIME_COLORS[regime]

        # Shaded std band
        ax.fill_between(rounds, rewards - stds, rewards + stds,
                        alpha=0.15, color=color)
        # MA-IPPO learning curve
        ax.plot(rounds, rewards, color=color, linewidth=2.0,
                marker='o', markersize=3.5, label='MA-IPPO')
        # Best baseline
        ax.axhline(best_bl, color='#e74c3c', linewidth=1.5,
                   linestyle='--', label=f'Best baseline ({best_bl:.1f})')

        # Best round marker
        best_idx = int(np.argmax(rewards))
        ax.scatter([rounds[best_idx]], [rewards[best_idx]],
                   color=color, s=80, zorder=5, edgecolors='black', linewidths=0.8)

        ax.set_title(REGIME_LABELS[regime], fontsize=11)
        ax.set_xlabel('Round', fontsize=10)
        ax.set_ylabel('Mean Reward', fontsize=10)
        ax.set_xticks(rounds[::2])
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.tight_layout()
    out_path = FIGURES_DIR / f'summary_learning_curves_{task}.png'
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def main() -> None:
    for task in ['make_cereal', 'latte_making']:
        print(f'Generating learning curves for {task}...')
        plot_task_learning_curves(task)
    print('Done.')


if __name__ == '__main__':
    main()
