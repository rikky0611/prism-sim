#!/usr/bin/env python3
"""Phase diagram visualization for 2D grid search results.

Reads grid_search_*.json and generates:
  - Categorical phase diagram (dominant behavior per grid point)
  - Continuous heatmaps (reward, narration, questions, failures)
  - Contour lines marking phase boundaries

Usage:
    python plot_phase_diagram.py \
        --results data/results/grid_search_make_cereal_step_transition_durable.json \
        [--output results/figures/phase_diagram_make_cereal.png]
        [--threshold 1.0]
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

# ── Behavioral categories ────────────────────────────────────────────────
#
# Category is determined by which communication channels are active
# (count ≥ threshold per episode in final eval).
#
CATEGORIES = [
    # (name,          label,              color,     priority)
    ('full_coop',  'Full cooperation',  '#7B1FA2',  7),   # narr+q+remind
    ('nar_q',      'Narrate + Ask',     '#E91E63',  6),   # narr+q
    ('q_remind',   'Ask + Remind',      '#009688',  5),   # q+remind
    ('nar_remind', 'Narrate + Remind',  '#FF5722',  4),   # narr+remind
    ('narration',  'Narration only',    '#FF9800',  3),
    ('question',   'Question only',     '#4CAF50',  2),
    ('remind',     'Remind only',       '#F44336',  1),
    ('silent',     'Silent',            '#1565C0',  0),
]

CAT_INDEX = {c[0]: i for i, c in enumerate(CATEGORIES)}
CAT_COLORS = [c[2] for c in CATEGORIES]
CAT_LABELS = [c[1] for c in CATEGORIES]


def classify(narr: float, q: float, interact: float, thresh: float) -> str:
    active_narr   = narr    >= thresh
    active_q      = q       >= thresh
    active_remind = interact >= thresh

    if active_narr and active_q and active_remind:
        return 'full_coop'
    if active_narr and active_q:
        return 'nar_q'
    if active_q and active_remind:
        return 'q_remind'
    if active_narr and active_remind:
        return 'nar_remind'
    if active_narr:
        return 'narration'
    if active_q:
        return 'question'
    if active_remind:
        return 'remind'
    return 'silent'


def load_grid(path: str):
    with open(path) as f:
        d = json.load(f)
    c_comm_vals = np.array(d['c_comm_vals'])
    c_fail_vals = np.array(d['c_fail_vals'])
    n_fail = len(c_fail_vals)
    n_comm = len(c_comm_vals)

    reward   = np.zeros((n_fail, n_comm))
    narr     = np.zeros((n_fail, n_comm))
    quest    = np.zeros((n_fail, n_comm))
    inter    = np.zeros((n_fail, n_comm))
    fail     = np.zeros((n_fail, n_comm))
    baseline = np.full((n_fail, n_comm), np.nan)

    for row in d['grid']:
        for cell in row:
            if cell is None:
                continue
            i = cell['i_fail']
            j = cell['i_comm']
            reward[i, j] = cell['mean_reward']
            narr[i, j]   = cell['mean_narrations']
            quest[i, j]  = cell['mean_questions']
            inter[i, j]  = cell['mean_interactions']
            fail[i, j]   = cell['mean_failures']
            b = cell.get('baseline_mean_reward')
            if b is not None:
                baseline[i, j] = b

    return d, c_comm_vals, c_fail_vals, reward, narr, quest, inter, fail, baseline


import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from _paper_export import add_paper_arg, export_to_paper


def plot_phase_diagram(results_path: str, output_path: str = None,
                       threshold: float = 1.0, paper: bool = False):
    d, c_comm, c_fail, reward, narr, quest, inter, failures, baseline = load_grid(results_path)

    task = d.get('task', 'unknown')
    decay = d.get('decay_regime', '')
    obs   = d.get('obs_regime', '')

    n_fail, n_comm = reward.shape

    has_baseline = bool(np.any(np.isfinite(baseline)))
    if has_baseline:
        reward_diff = reward - baseline
    else:
        reward_diff = None
        print("[warn] no baseline_mean_reward in JSON — "
              "run backfill_grid_baselines.py to enable ΔReward panel. "
              "Falling back to 5-panel layout.")

    # ── Classify each cell ──────────────────────────────────────────────
    cat_idx = np.zeros((n_fail, n_comm), dtype=int)
    for i in range(n_fail):
        for j in range(n_comm):
            cat = classify(narr[i, j], quest[i, j], inter[i, j], threshold)
            cat_idx[i, j] = CAT_INDEX[cat]

    # ── Build categorical colormap ───────────────────────────────────────
    cmap_cat = mcolors.ListedColormap(CAT_COLORS)
    norm_cat = mcolors.BoundaryNorm(np.arange(-0.5, len(CATEGORIES)), len(CATEGORIES))

    # ── Figure setup ────────────────────────────────────────────────────
    n_panels = 6 if has_baseline else 5
    fig, axes = plt.subplots(1, n_panels, figsize=(4.4 * n_panels, 5.5))
    fig.patch.set_facecolor('white')

    # Log-spaced tick positions
    x_ticks = np.arange(n_comm)
    y_ticks = np.arange(n_fail)
    x_labels = [f'{v:.2f}' if v < 1 else f'{v:.1f}' for v in c_comm]
    y_labels = [f'{v:.1f}' for v in c_fail]

    def setup_ax(ax, title):
        ax.set_xticks(x_ticks[::max(1, n_comm//6)])
        ax.set_xticklabels([x_labels[k] for k in x_ticks[::max(1, n_comm//6)]],
                           fontsize=8, rotation=45)
        ax.set_yticks(y_ticks[::max(1, n_fail//6)])
        ax.set_yticklabels([y_labels[k] for k in y_ticks[::max(1, n_fail//6)]],
                           fontsize=8)
        ax.set_xlabel('c_comm  (= c_nar = c_q)', fontsize=10)
        ax.set_ylabel('c_fail_scale', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_aspect('auto')

    # ── Panel 1: Phase diagram (categorical) ────────────────────────────
    ax = axes[0]
    im = ax.imshow(cat_idx, origin='lower', aspect='auto',
                   cmap=cmap_cat, norm=norm_cat,
                   interpolation='nearest')

    # Contour lines at category boundaries
    if n_fail > 2 and n_comm > 2:
        try:
            ax.contour(cat_idx, levels=np.arange(0.5, len(CATEGORIES)),
                       colors='white', linewidths=0.8, alpha=0.7)
        except Exception:
            pass

    setup_ax(ax, f'Phase Diagram  (thresh={threshold})')

    # Legend
    patches = [Patch(color=CAT_COLORS[i], label=CAT_LABELS[i])
               for i in range(len(CATEGORIES))
               if np.any(cat_idx == i)]
    ax.legend(handles=patches, loc='upper left', fontsize=7,
              framealpha=0.85, edgecolor='#ccc')

    # ── Panel 2: Reward ─────────────────────────────────────────────────
    ax = axes[1]
    vmax = max(abs(reward.min()), abs(reward.max()))
    im2 = ax.imshow(reward, origin='lower', aspect='auto',
                    cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    setup_ax(ax, 'Mean Reward')

    # ── Panel 3 (optional): ΔReward = RL − no-system baseline ────────────
    next_idx = 2
    if has_baseline:
        ax = axes[next_idx]
        finite = np.isfinite(reward_diff)
        vmax_d = float(np.nanmax(np.abs(reward_diff[finite]))) if finite.any() else 1.0
        if vmax_d == 0.0:
            vmax_d = 1.0
        im_d = ax.imshow(reward_diff, origin='lower', aspect='auto',
                         cmap='RdBu_r', vmin=-vmax_d, vmax=vmax_d)
        plt.colorbar(im_d, ax=ax, fraction=0.046, pad=0.04)
        # 0-contour: boundary where RL ties the no-system baseline.
        try:
            ax.contour(reward_diff, levels=[0.0],
                       colors='k', linewidths=1.0, alpha=0.6)
        except Exception:
            pass
        setup_ax(ax, 'ΔReward (RL − no-system)')
        next_idx += 1

    # ── Panel: Narrations ───────────────────────────────────────────────
    ax = axes[next_idx]
    im3 = ax.imshow(narr, origin='lower', aspect='auto',
                    cmap='YlOrRd', vmin=0)
    plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
    ax.contour(narr, levels=[threshold], colors='white',
               linewidths=1.5, linestyles='--', alpha=0.9)
    setup_ax(ax, 'Narrations / episode')
    next_idx += 1

    # ── Panel: Questions ────────────────────────────────────────────────
    ax = axes[next_idx]
    im4 = ax.imshow(quest, origin='lower', aspect='auto',
                    cmap='YlGn', vmin=0)
    plt.colorbar(im4, ax=ax, fraction=0.046, pad=0.04)
    ax.contour(quest, levels=[threshold], colors='white',
               linewidths=1.5, linestyles='--', alpha=0.9)
    setup_ax(ax, 'Questions / episode')
    next_idx += 1

    # ── Panel: Failures ─────────────────────────────────────────────────
    ax = axes[next_idx]
    im5 = ax.imshow(failures, origin='lower', aspect='auto',
                    cmap='Reds', vmin=0)
    plt.colorbar(im5, ax=ax, fraction=0.046, pad=0.04)
    setup_ax(ax, 'Failures / episode')

    # ── Suptitle ─────────────────────────────────────────────────────────
    fig.suptitle(
        f'Phase Diagram: {task}  |  decay={decay}  obs={obs}  '
        f'c_remind={d.get("c_remind",1.0):.1f}  c_confirm={d.get("c_confirm",1.0):.1f}\n'
        f'Grid: {n_comm} × {n_fail}  |  '
        f'rounds={d.get("n_rounds")}  steps/round={d.get("steps_per_round")}',
        fontsize=11, fontweight='bold', y=1.02,
    )
    fig.subplots_adjust(wspace=0.38)

    if output_path is None:
        output_path = str(
            PROJECT_ROOT / 'results' / 'figures' /
            f'phase_diagram_{task}_{decay}_{obs}.png'
        )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {output_path}')
    export_to_paper(output_path, paper=paper)
    plt.close(fig)

    # ── Print category counts ────────────────────────────────────────────
    print(f'\nBehavior distribution (threshold={threshold}):')
    for i, (name, label, color, _) in enumerate(CATEGORIES):
        count = np.sum(cat_idx == i)
        if count > 0:
            pct = 100 * count / (n_fail * n_comm)
            print(f'  {label:22s}  {count:3d} / {n_fail*n_comm}  ({pct:.1f}%)')


def main():
    parser = argparse.ArgumentParser(description='Plot phase diagram from grid search')
    parser.add_argument('--results', required=True,
                        help='Path to grid_search_*.json')
    parser.add_argument('--output', default=None,
                        help='Output PNG path (auto-generated if omitted)')
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='Communication count threshold for behavior classification')
    add_paper_arg(parser)
    args = parser.parse_args()
    plot_phase_diagram(args.results, args.output, args.threshold, paper=args.paper)


if __name__ == '__main__':
    main()
