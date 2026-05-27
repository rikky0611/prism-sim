#!/usr/bin/env python3
"""Trajectory anatomy plot for representative regimes (E5).

Loads trained MA-IPPO models for a curated set of regimes that span the
phase diagram, rolls out one episode each, and plots a tick-aligned panel
per regime showing:
  - true step (top trace)
  - memory[current critical step]
  - obs_noise_state
  - human action events (silent / narrate / question_*)
  - assistant action events (silent / confirm / remind_*)

The goal is to make emergent strategies inspectable: in the human-led
phase, narration fires early in each step to keep noise low; in the
assistant-led phase, reminders cluster before critical steps; in mixed
regimes both happen.

Usage:
    python plot_trajectory_anatomy.py [--task make_cereal] [--seed 0]
"""

import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'experiments'))

from task_definitions import get_task_definition
from regime_definitions import build_params_grid, build_params_asymmetric
from ma_procedure_assistant_sim import MAProcedureAssistantEnv
from stable_baselines3 import PPO


# Curated regimes spanning the cost-asymmetry plane.
# Selection comes from the asymmetric grid (compatible with current v3 obs).
# Cells will be filled in by the asym grid search; here we use the 4 corners.
def make_regimes(task_def, c_fail_scale: float = 15.0):
    """Cells chosen from the E1 6×6 grid to span the four phases cleanly."""
    return [
        ('Silent phase  (c_nar=5.0,  c_remind=5.0)',
         'asym_cn5.0000_cr5.0000_seed0',
         build_params_asymmetric,
         dict(c_fail_scale=c_fail_scale, c_nar=5.0, c_remind=5.0,
              decay_regime='step_transition', obs_regime='durable')),
        ('Human-led phase  (c_nar=0.13,  c_remind=5.0)',
         'asym_cn0.1256_cr5.0000_seed0',
         build_params_asymmetric,
         dict(c_fail_scale=c_fail_scale, c_nar=0.1256, c_remind=5.0,
              decay_regime='step_transition', obs_regime='durable')),
        ('Assistant-led phase  (c_nar=1.99,  c_remind=0.13)',
         'asym_cn1.9905_cr0.1256_seed0',
         build_params_asymmetric,
         dict(c_fail_scale=c_fail_scale, c_nar=1.9905, c_remind=0.1256,
              decay_regime='step_transition', obs_regime='durable')),
        ('Mixed phase  (c_nar=0.13,  c_remind=0.05)',
         'asym_cn0.1256_cr0.0500_seed0',
         build_params_asymmetric,
         dict(c_fail_scale=c_fail_scale, c_nar=0.1256, c_remind=0.05,
              decay_regime='step_transition', obs_regime='durable')),
    ]


# Action color/marker palette
C_NAR  = '#0173b2'
C_QUE  = '#029e73'
C_REM  = '#de8f05'
C_CON  = '#cc78bc'


def rollout(ma_env, human_model, assistant_model, max_ticks=600):
    """Run a single deterministic episode and collect tick-aligned traces."""
    h_obs_dict, a_obs_dict = ma_env.reset()
    n_steps = ma_env.n_steps

    # Identify a critical step for memory trace
    crit_idx = ma_env.critical_steps[0] if ma_env.critical_steps else 0

    def conv_h(d):
        return np.array([
            d['current_identity'], d['tau'], d['memory_current'],
            d['assistant_last_action'], d['obs_noise_state'],
        ], dtype=np.float32)

    def conv_a(d):
        return np.concatenate([
            d['step_belief'].astype(np.float32),
            d['expected_tau'].astype(np.float32),
            d['memory_estimate_critical'].astype(np.float32),
            np.array([d['human_last_action']], dtype=np.float32),
        ])

    rec = {
        'tick': [], 'true_step': [], 'mem_crit': [], 'obs_noise': [],
        'h_action': [], 'a_action': [], 'failure_tick': [], 'completed_tick': [],
    }

    while not ma_env.ma_state.is_done and len(rec['tick']) < max_ticks:
        h_obs = conv_h(h_obs_dict)
        a_obs = conv_a(a_obs_dict)
        h_action, _ = human_model.predict(h_obs, deterministic=True)
        a_action, _ = assistant_model.predict(a_obs, deterministic=True)

        h_obs_dict, a_obs_dict, reward, done, info = ma_env.step(
            int(h_action), int(a_action)
        )

        rec['tick'].append(len(rec['tick']))
        rec['true_step'].append(info['true_step'])
        rec['mem_crit'].append(float(info['memory'][crit_idx]) if crit_idx < len(info['memory']) else 0.0)
        rec['obs_noise'].append(float(info['obs_noise_state']))
        rec['h_action'].append(int(h_action))
        rec['a_action'].append(int(a_action))
        if info.get('failure'):
            rec['failure_tick'].append(rec['tick'][-1])
        if info.get('step_completed'):
            rec['completed_tick'].append(rec['tick'][-1])

    rec['crit_idx'] = crit_idx
    rec['n_steps'] = n_steps
    return rec


def classify_action(action_id: int, env: MAProcedureAssistantEnv, is_human: bool):
    """Map action id to ('silent'|'general'|'targeted', step_idx_or_None)."""
    if action_id == 0:
        return ('silent', None)
    if action_id == 1:
        return ('general', None)  # narrate (human) or confirm (assistant)
    # step-targeted: question_i (human) or remind_i (assistant)
    j = action_id - 2
    if is_human:
        if j < len(env.critical_steps):
            return ('targeted', env.critical_steps[j])
    else:
        if j < len(env.critical_steps):
            return ('targeted', env.critical_steps[j])
    return ('silent', None)


import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from _paper_export import add_paper_arg, export_to_paper


def plot_trajectories(task_name='make_cereal', seed=0,
                      output_path=None, paper=False):
    task_def = get_task_definition(task_name)
    regimes = make_regimes(task_def)

    fig, axes = plt.subplots(len(regimes), 1,
                             figsize=(15, 3.0 * len(regimes)),
                             sharex=False)
    fig.patch.set_facecolor('white')

    for ax, (label, model_subdir, builder, kwargs) in zip(axes, regimes):
        params = builder(task_def, **kwargs)
        ma_env = MAProcedureAssistantEnv(params, task_def)

        model_dir = PROJECT_ROOT / 'models' / 'ma_ippo' / task_name / model_subdir
        if not model_dir.exists():
            ax.set_title(f'{label}  [models missing: {model_subdir}]', fontsize=10, color='red')
            ax.axis('off')
            continue

        try:
            human_model = PPO.load(str(model_dir / 'human_model_best'))
            assistant_model = PPO.load(str(model_dir / 'assistant_model_best'))
        except Exception as e:
            ax.set_title(f'{label}  [load error: {e}]', fontsize=10, color='red')
            ax.axis('off')
            continue

        np.random.seed(seed)
        random.seed(seed)
        rec = rollout(ma_env, human_model, assistant_model)

        ticks = np.array(rec['tick'])
        if len(ticks) == 0:
            ax.set_title(f'{label}  [empty rollout]', fontsize=10, color='red')
            continue

        # Three traces sharing a single panel: step / memory / obs_noise
        ax.plot(ticks, rec['true_step'], color='black', lw=1.5, label='true step')
        ax.set_ylim(-0.5, rec['n_steps'] + 0.5)
        ax.set_ylabel('step', fontsize=9)

        ax_mem = ax.twinx()
        ax_mem.plot(ticks, rec['mem_crit'], color='#9C27B0', lw=1.2, alpha=0.8,
                    label=f'mem[crit={rec["crit_idx"]}]')
        ax_mem.plot(ticks, rec['obs_noise'], color='#FFB300', lw=1.2, alpha=0.8,
                    linestyle='--', label='obs_noise')
        ax_mem.set_ylim(-0.05, 1.05)
        ax_mem.set_ylabel('memory / noise', fontsize=8, color='#666')
        ax_mem.tick_params(axis='y', labelsize=7, colors='#666')

        # Mark step completions and failures
        for t in rec['completed_tick']:
            ax.axvline(t, color='#1E88E5', linewidth=0.5, alpha=0.25)
        for t in rec['failure_tick']:
            ax.axvline(t, color='red', linewidth=1.2, alpha=0.5)

        # Action event markers — plot as small ticks on the y=-0.3 (human) and y=-0.6 (assistant) lines
        h_y = -0.3
        a_y = -0.7
        ax.set_ylim(-1.0, rec['n_steps'] + 0.5)
        # human
        for t, act in zip(ticks, rec['h_action']):
            kind, step_idx = classify_action(act, ma_env, is_human=True)
            if kind == 'silent':
                continue
            if kind == 'general':
                ax.scatter([t], [h_y], marker='|', color=C_NAR, s=80, linewidths=2, zorder=5)
            else:
                ax.scatter([t], [h_y], marker='v', color=C_QUE, s=40, zorder=5)
        # assistant
        for t, act in zip(ticks, rec['a_action']):
            kind, step_idx = classify_action(act, ma_env, is_human=False)
            if kind == 'silent':
                continue
            if kind == 'general':
                ax.scatter([t], [a_y], marker='|', color=C_CON, s=80, linewidths=2, zorder=5)
            else:
                ax.scatter([t], [a_y], marker='^', color=C_REM, s=40, zorder=5)

        ax.text(-0.005 * max(1, ticks[-1]), h_y, 'human:', fontsize=7, color='#666',
                ha='right', va='center')
        ax.text(-0.005 * max(1, ticks[-1]), a_y, 'asst.:', fontsize=7, color='#666',
                ha='right', va='center')

        ax.set_title(label, fontsize=11, fontweight='bold', loc='left')
        ax.set_xlabel('tick', fontsize=9)
        ax.grid(True, alpha=0.2)

        # Legend (only on first subplot)
        if ax is axes[0]:
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
            legend_items = [
                Line2D([0], [0], color='black', lw=1.5, label='true step'),
                Line2D([0], [0], color='#9C27B0', lw=1.2, label='mem[crit]'),
                Line2D([0], [0], color='#FFB300', lw=1.2, linestyle='--', label='obs_noise'),
                Line2D([0], [0], marker='|', color=C_NAR, lw=0, markersize=10, label='narrate'),
                Line2D([0], [0], marker='v', color=C_QUE, lw=0, markersize=7, label='question_i'),
                Line2D([0], [0], marker='|', color=C_CON, lw=0, markersize=10, label='confirm'),
                Line2D([0], [0], marker='^', color=C_REM, lw=0, markersize=7, label='remind_i'),
            ]
            ax.legend(handles=legend_items, loc='upper right', fontsize=7,
                      ncol=4, framealpha=0.85)

    fig.suptitle(
        f'Trajectory anatomy — emergent communicative strategies across phases  '
        f'(task: {task_name}, deterministic rollout, seed={seed})',
        fontsize=12, fontweight='bold', y=1.00,
    )
    fig.tight_layout(rect=[0.02, 0, 1, 0.97])

    if output_path is None:
        output_path = str(PROJECT_ROOT / 'results' / 'figures' /
                          f'trajectory_anatomy_{task_name}_seed{seed}.png')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {output_path}')
    export_to_paper(output_path, paper=paper)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='make_cereal')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', default=None)
    add_paper_arg(parser)
    args = parser.parse_args()
    plot_trajectories(args.task, args.seed, args.output, paper=args.paper)


if __name__ == '__main__':
    main()
