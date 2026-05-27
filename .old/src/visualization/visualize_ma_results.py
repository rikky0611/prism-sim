"""
Multi-Agent IPPO Results Visualization

Generates three figures from obs_noise comparison experiments:

  Figure 1: Training Curves
    - 5 metrics × 3 regimes, two noise conditions overlaid
    - Output: results/figures/ma_training_curves_noise_comparison.png

  Figure 2: Final Metrics Bar Chart
    - Reward and failures grouped by noise condition and baseline
    - Output: results/figures/ma_final_metrics_noise_comparison.png

  Figure 3: Episode Timeline
    - Horizontal axis = tick, vertical axis = step index
    - Colored vertical lines mark interaction events
    - Output: results/figures/ma_episode_timeline_noise_comparison.png

Usage:
  python3 src/visualization/visualize_ma_results.py
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Any, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'results'
FIGURES_DIR = PROJECT_ROOT / 'results' / 'figures'


# ============================================================================
# STYLE CONSTANTS
# ============================================================================
REGIME_LABELS = {
    'extremely_low': 'Extremely Low\n(c_fail × 2)',
    'balanced': 'Balanced\n(c_fail × 15)',
    'extremely_high': 'Extremely High\n(c_fail × 50)',
}
REGIME_ORDER = ['extremely_low', 'balanced', 'extremely_high']

NOISE_STYLES = {
    'noise02': {'label': 'obs_noise = 0.20', 'ls': '-',  'lw': 2.0, 'alpha': 1.0},
    'noise05': {'label': 'obs_noise = 0.50', 'ls': '--', 'lw': 2.0, 'alpha': 0.85},
}

METRIC_STYLES = {
    'mean_reward':       {'color': '#1F77B4', 'label': 'Reward'},
    'mean_failures':     {'color': '#D62728', 'label': 'Failures'},
    'mean_narrations':   {'color': '#9467BD', 'label': 'Narrations (H)'},
    'mean_questions':    {'color': '#2CA02C', 'label': 'Questions (H)'},
    'mean_interactions': {'color': '#FF7F0E', 'label': 'Reminders (A)'},
}
METRIC_ORDER = list(METRIC_STYLES.keys())

# Timeline interaction colors
TIMELINE_COLORS = {
    'narrate':  {'color': '#4472C4', 'label': 'Human: narrate',   'ls': '--', 'lw': 1.6},
    'question': {'color': '#70AD47', 'label': 'Human: question',  'ls': ':',  'lw': 1.6},
    'remind':   {'color': '#FF4444', 'label': 'Asst: remind',     'ls': '-',  'lw': 1.4},
    'confirm':  {'color': '#A5A5A5', 'label': 'Asst: confirm',    'ls': '--', 'lw': 1.2},
}


# ============================================================================
# DATA LOADING
# ============================================================================
def load_results(noise_label: str, task: str = 'make_cereal') -> Optional[Dict]:
    """Load JSON results for a given noise condition."""
    path = DATA_DIR / f'ma_v2_{noise_label}_{task}_all.json'
    if not path.exists():
        print(f'  [WARN] Missing: {path}')
        return None
    with open(path) as f:
        return json.load(f)


# ============================================================================
# FIGURE 1: TRAINING CURVES
# ============================================================================
def plot_training_curves(noise_results: Dict[str, Dict]):
    """5 metrics × 3 regimes subplot grid.

    Two noise conditions overlaid per subplot (solid vs dashed).
    """
    n_metrics = len(METRIC_ORDER)
    n_regimes = len(REGIME_ORDER)

    fig, axes = plt.subplots(
        n_metrics, n_regimes,
        figsize=(5 * n_regimes, 3.2 * n_metrics),
        sharex=False,
    )
    fig.suptitle('MA-IPPO Training Curves: obs_noise Comparison',
                 fontsize=15, weight='bold', y=1.01)

    for col, regime in enumerate(REGIME_ORDER):
        for row, metric_key in enumerate(METRIC_ORDER):
            ax = axes[row, col]
            style_m = METRIC_STYLES[metric_key]

            for noise_label, data in noise_results.items():
                if data is None:
                    continue
                regime_data = data['regimes'].get(regime, {})
                rounds = regime_data.get('ma_ippo', {}).get('training_log', [])
                if not rounds:
                    continue

                xs = [r['round'] for r in rounds]
                ys = [r.get(metric_key, 0.0) for r in rounds]
                ns = NOISE_STYLES[noise_label]

                ax.plot(xs, ys,
                        color=style_m['color'],
                        ls=ns['ls'], lw=ns['lw'], alpha=ns['alpha'],
                        label=ns['label'])

            # Baseline reference line for reward and failures
            if metric_key == 'mean_reward':
                regime_data = next(
                    (v['regimes'].get(regime, {}) for v in noise_results.values()
                     if v is not None), {}
                )
                bl = regime_data.get('baselines', {}).get('both_silent', {})
                if bl:
                    ax.axhline(bl.get('mean_reward', 0), color='gray',
                               ls=':', lw=1.2, alpha=0.6, label='Baseline (silent)')
            elif metric_key == 'mean_failures':
                regime_data = next(
                    (v['regimes'].get(regime, {}) for v in noise_results.values()
                     if v is not None), {}
                )
                bl = regime_data.get('baselines', {}).get('both_silent', {})
                if bl:
                    ax.axhline(bl.get('mean_failures', 0), color='gray',
                               ls=':', lw=1.2, alpha=0.6, label='Baseline (silent)')

            # Axis labels
            if row == 0:
                ax.set_title(REGIME_LABELS[regime], fontsize=11, weight='bold')
            if col == 0:
                ax.set_ylabel(style_m['label'], fontsize=10)
            if row == n_metrics - 1:
                ax.set_xlabel('Training Round', fontsize=10)

            ax.set_xlim(0.5, None)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc='lower right')

    # Global legend for noise conditions
    handles = [
        mpatches.Patch(color='#555555', label=s['label'],
                       linestyle=s['ls'], linewidth=s['lw'])
        for _, s in NOISE_STYLES.items()
    ]
    # Use lines instead of patches for linestyle legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color='#555555', ls=s['ls'], lw=s['lw'], label=s['label'])
        for s in NOISE_STYLES.values()
    ]
    fig.legend(
        handles=legend_handles, loc='lower center',
        ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.02),
        frameon=True,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / 'ma_training_curves_noise_comparison.png'
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


# ============================================================================
# FIGURE 2: FINAL METRICS BAR CHART
# ============================================================================
def plot_final_metrics_bars(noise_results: Dict[str, Dict]):
    """Grouped bar chart: reward and failures, by regime, grouped by condition."""
    n_regimes = len(REGIME_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('MA-IPPO Final Evaluation: obs_noise Comparison',
                 fontsize=14, weight='bold')

    metrics_to_plot = [
        ('mean_reward',   'Mean Reward',   '#1F77B4'),
        ('mean_failures', 'Mean Failures', '#D62728'),
    ]

    for ax, (metric_key, metric_label, bar_color) in zip(axes, metrics_to_plot):
        # For each regime, collect bars:
        # noise02_final, noise05_final, baseline_silent
        noise_labels = list(noise_results.keys())
        n_groups = n_regimes
        n_bars = len(noise_labels) + 1  # +1 for baseline
        width = 0.22
        x = np.arange(n_groups)

        bars_data = []  # list of (values, label, color, hatch)

        for i, noise_label in enumerate(noise_labels):
            data = noise_results[noise_label]
            vals = []
            for regime in REGIME_ORDER:
                rd = data['regimes'].get(regime, {}) if data else {}
                v = rd.get('ma_ippo', {}).get('final_eval', {}).get(metric_key, 0.0)
                vals.append(v)
            ns = NOISE_STYLES[noise_label]
            bars_data.append((vals, f'MA-IPPO ({ns["label"]})', bar_color, '' if i == 0 else '//'))

        # Baseline (both silent) from noise02 (same environment baseline regardless of noise)
        bl_vals = []
        for regime in REGIME_ORDER:
            data_ref = next((v for v in noise_results.values() if v is not None), None)
            if data_ref:
                bl = data_ref['regimes'].get(regime, {}).get('baselines', {}).get('both_silent', {})
                bl_vals.append(bl.get(metric_key, 0.0))
            else:
                bl_vals.append(0.0)
        bars_data.append((bl_vals, 'Baseline (silent)', '#7F7F7F', 'xx'))

        offset = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * width

        for bi, (vals, blabel, color, hatch) in enumerate(bars_data):
            bars = ax.bar(
                x + offset[bi], vals, width,
                label=blabel, color=color, hatch=hatch,
                alpha=0.85, edgecolor='white',
            )

        ax.set_xticks(x)
        ax.set_xticklabels([REGIME_LABELS[r].replace('\n', '\n') for r in REGIME_ORDER],
                           fontsize=9)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(metric_label, fontsize=12, weight='bold')
        ax.axhline(0, color='black', lw=0.8)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    out = FIGURES_DIR / 'ma_final_metrics_noise_comparison.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


# ============================================================================
# FIGURE 3: EPISODE TIMELINE
# ============================================================================
def _classify_action(h_act: int, a_act: int) -> List[tuple]:
    """Return list of (event_type, target_j_or_None) for the tick.

    target_j: index into critical_steps list (for remind/question), else None.
    """
    events = []
    if h_act == 1:
        events.append(('narrate', None))
    elif h_act >= 2:
        events.append(('question', h_act - 2))
    if a_act == 1:
        events.append(('confirm', None))
    elif a_act >= 2:
        events.append(('remind', a_act - 2))
    return events


def plot_episode_timeline(
    noise_results: Dict[str, Dict],
    regime_order: List[str] = REGIME_ORDER,
    trace_idx: int = 0,
):
    """2 noise conditions × 3 regimes grid.

    Each cell shows one representative episode:
      - Black staircase: true step progress
      - Gray line: assistant's step_belief argmax
      - Colored vertical lines: interaction events
    """
    noise_labels = list(noise_results.keys())
    n_rows = len(noise_labels)
    n_cols = len(regime_order)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6.5 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )
    fig.suptitle('Episode Trajectory with Interaction Events',
                 fontsize=15, weight='bold', y=1.01)

    for row, noise_label in enumerate(noise_labels):
        data = noise_results[noise_label]
        ns_style = NOISE_STYLES[noise_label]

        for col, regime in enumerate(regime_order):
            ax = axes[row, col]

            # Column / row headers
            if row == 0:
                ax.set_title(REGIME_LABELS[regime], fontsize=11, weight='bold')
            if col == 0:
                ax.set_ylabel(
                    f'{ns_style["label"]}\nStep Index',
                    fontsize=10,
                )

            if data is None:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', va='center', color='gray')
                continue

            rd = data['regimes'].get(regime, {})
            traces = rd.get('ma_ippo', {}).get('episode_traces', [])
            step_names = rd.get('step_names', [])
            n_steps = len(step_names) if step_names else 8
            n_critical = rd.get('n_critical', 0)
            critical_steps = rd.get('critical_steps', [])

            if not traces or trace_idx >= len(traces):
                ax.text(0.5, 0.5, 'No trace data', transform=ax.transAxes,
                        ha='center', va='center', color='gray')
                continue

            trace = traces[trace_idx]
            ticks_arr = [r['tick'] for r in trace]
            steps_arr = [r['true_step'] for r in trace]
            belief_arr = [r['step_belief_argmax'] for r in trace]

            # --- Background: light horizontal bands per step ---
            for s_idx in range(n_steps):
                color = '#F5F5DC' if s_idx % 2 == 0 else '#EFEFEF'
                ax.axhspan(s_idx - 0.5, s_idx + 0.5, color=color, alpha=0.5, zorder=0)

            # --- True step staircase ---
            ax.step(ticks_arr, steps_arr, where='post',
                    color='black', lw=2.0, label='True step', zorder=5)

            # --- Belief argmax (gray thin line) ---
            ax.plot(ticks_arr, belief_arr,
                    color='#AAAAAA', lw=1.0, ls='--', alpha=0.7,
                    label="Belief argmax", zorder=4)

            # --- Interaction events ---
            for rec in trace:
                events = _classify_action(rec['h_action'], rec['a_action'])
                for etype, target_j in events:
                    tc = TIMELINE_COLORS[etype]
                    if etype in ('narrate', 'confirm'):
                        # Full-height vertical line
                        ax.axvline(
                            rec['tick'], color=tc['color'],
                            ls=tc['ls'], lw=tc['lw'], alpha=0.55, zorder=3,
                        )
                    else:
                        # Marker at target step y-position
                        if target_j is not None and target_j < len(critical_steps):
                            target_y = critical_steps[target_j]
                        else:
                            target_y = rec['true_step']
                        marker = 'v' if etype == 'remind' else '^'
                        ax.scatter(
                            rec['tick'], target_y,
                            marker=marker, s=90,
                            color=tc['color'], edgecolors='white', linewidths=0.8,
                            zorder=6, alpha=0.9,
                        )

            # Y-axis: step names
            y_ticks = list(range(n_steps))
            if step_names:
                y_labels = [n[:14] for n in step_names[:n_steps]]
            else:
                y_labels = [str(i) for i in y_ticks]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels, fontsize=7)
            ax.set_ylim(-0.6, n_steps - 0.4)

            ax.set_xlabel('Tick', fontsize=9)
            ax.grid(True, axis='x', alpha=0.25)
            ax.tick_params(axis='x', labelsize=8)

    # Legend (shared, bottom)
    legend_elements = [
        plt.Line2D([0], [0], color='black', lw=2.0, label='True step'),
        plt.Line2D([0], [0], color='#AAAAAA', lw=1.0, ls='--', label='Belief argmax'),
        plt.Line2D([0], [0], color=TIMELINE_COLORS['narrate']['color'],
                   lw=TIMELINE_COLORS['narrate']['lw'],
                   ls=TIMELINE_COLORS['narrate']['ls'],
                   label=TIMELINE_COLORS['narrate']['label']),
        plt.Line2D([0], [0], color=TIMELINE_COLORS['confirm']['color'],
                   lw=TIMELINE_COLORS['confirm']['lw'],
                   ls=TIMELINE_COLORS['confirm']['ls'],
                   label=TIMELINE_COLORS['confirm']['label']),
        plt.Line2D([0], [0], marker='^', color='w',
                   markerfacecolor=TIMELINE_COLORS['question']['color'],
                   markersize=9, label=TIMELINE_COLORS['question']['label'] + ' (target step ▲)'),
        plt.Line2D([0], [0], marker='v', color='w',
                   markerfacecolor=TIMELINE_COLORS['remind']['color'],
                   markersize=9, label=TIMELINE_COLORS['remind']['label'] + ' (target step ▼)'),
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center', ncol=len(legend_elements),
        fontsize=9, bbox_to_anchor=(0.5, -0.04), frameon=True,
    )

    fig.tight_layout()
    out = FIGURES_DIR / 'ma_episode_timeline_noise_comparison.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


# ============================================================================
# MAIN
# ============================================================================
def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading results...', flush=True)
    noise_results: Dict[str, Optional[Dict]] = {}
    for label in ['noise02', 'noise05']:
        data = load_results(label)
        if data is not None:
            print(f'  Loaded: {label}  regimes={list(data["regimes"].keys())}')
        noise_results[label] = data

    if all(v is None for v in noise_results.values()):
        print('ERROR: No result files found. Run run_noise_comparison.py first.')
        return

    print('\n--- Figure 1: Training Curves ---', flush=True)
    plot_training_curves(noise_results)

    print('\n--- Figure 2: Final Metrics Bar Chart ---', flush=True)
    plot_final_metrics_bars(noise_results)

    print('\n--- Figure 3: Episode Timeline ---', flush=True)
    plot_episode_timeline(noise_results)

    print('\nAll figures saved to:', FIGURES_DIR)


if __name__ == '__main__':
    main()
