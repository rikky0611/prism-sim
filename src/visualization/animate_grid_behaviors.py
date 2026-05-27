#!/usr/bin/env python3
"""
Animate representative episodes for each emergent behavior type
from the grid search phase diagram.

Picks one representative grid point per behavior type, loads
the trained model, records an episode, and saves an MP4/GIF.

Usage:
    python3 src/visualization/animate_grid_behaviors.py \
        --results data/results/grid_search_make_cereal_step_transition_durable.json \
        --seed 42

Output:
    results/videos/grid_phase_{behavior_type}.mp4
"""
import sys
import json
import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training"))

from task_definitions import get_task_definition
from regime_definitions import build_params_grid
from ma_procedure_assistant_sim import MAProcedureAssistantEnv, MASimulationParams
from stable_baselines3 import PPO

# Reuse animation infrastructure from animate_ma_episode
from animate_ma_episode import (
    TickRecord, EpisodeRecorder, MAEpisodeAnimator,
)


# ============================================================================
# Behavior classification (mirrors plot_phase_diagram.py)
# ============================================================================
BEHAVIOR_LABELS = {
    'full_coop':  'Full Cooperation (narr + q + remind)',
    'nar_q':      'Narrate + Question',
    'q_remind':   'Question + Remind',
    'nar_remind': 'Narrate + Remind',
    'narration':  'Narration Only',
    'question':   'Question Only',
    'remind':     'Remind Only',
    'silent':     'Silent',
}

BEHAVIOR_ORDER = [
    'silent', 'question', 'remind', 'narration',
    'q_remind', 'nar_q', 'nar_remind', 'full_coop',
]


def classify_behavior(narr, q, interact, threshold=1.0):
    has_n = narr > threshold
    has_q = q > threshold
    has_r = interact > threshold
    if has_n and has_q and has_r:
        return 'full_coop'
    elif has_n and has_q:
        return 'nar_q'
    elif has_q and has_r:
        return 'q_remind'
    elif has_n and has_r:
        return 'nar_remind'
    elif has_n:
        return 'narration'
    elif has_q:
        return 'question'
    elif has_r:
        return 'remind'
    else:
        return 'silent'


def find_representatives(grid_data):
    """Find representative grid point for each behavior type."""
    types = defaultdict(list)

    for row in grid_data['grid']:
        for cell in row:
            cf = cell['c_fail_scale']
            cc = cell['c_comm']
            narr = cell['mean_narrations']
            q = cell['mean_questions']
            interact = cell['mean_interactions']
            reward = cell['mean_reward']
            fail = cell['mean_failures']

            btype = classify_behavior(narr, q, interact)
            types[btype].append({
                'c_fail': cf, 'c_comm': cc,
                'reward': reward, 'narr': narr, 'q': q,
                'interact': interact, 'fail': fail,
            })

    # Pick representative: median by reward
    reps = {}
    for btype in BEHAVIOR_ORDER:
        pts = types.get(btype, [])
        if not pts:
            continue
        pts.sort(key=lambda x: x['reward'])
        rep = pts[len(pts) // 2]
        reps[btype] = rep
        print(f"  {btype:15s} ({len(pts):3d} pts) → "
              f"cf={rep['c_fail']:6.1f} cc={rep['c_comm']:.4f}  "
              f"r={rep['reward']:+.2f} narr={rep['narr']:.1f} "
              f"q={rep['q']:.1f} int={rep['interact']:.1f} fail={rep['fail']:.2f}")

    return reps


# ============================================================================
# Grid-aware episode recorder
# ============================================================================
class GridEpisodeRecorder:
    """Record an episode using a grid-search trained model."""

    def run_episode(self, task_name, c_fail, c_comm,
                    decay_regime='step_transition', obs_regime='durable',
                    seed=42, deterministic=False):
        task_def = get_task_definition(task_name)
        params = build_params_grid(
            task_def, c_fail_scale=c_fail, c_comm=c_comm,
            decay_regime=decay_regime, obs_regime=obs_regime,
        )

        ma_env = MAProcedureAssistantEnv(params, task_def)

        # Load models from grid directory
        regime_dir = f"cf{c_fail:.2f}_cc{c_comm:.4f}"
        model_dir = PROJECT_ROOT / "models" / "ma_ippo" / task_name / regime_dir

        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found: {model_dir}")

        human_model = PPO.load(str(model_dir / "human_model_best"))
        assistant_model = PPO.load(str(model_dir / "assistant_model_best"))

        # Run episode
        np.random.seed(seed)
        random.seed(seed)
        h_obs_dict, a_obs_dict = ma_env.reset()

        def convert_h(obs_dict):
            return np.array([
                obs_dict['current_identity'], obs_dict['tau'],
                obs_dict['memory_current'], obs_dict['assistant_last_action'],
                obs_dict['obs_noise_state'],
            ], dtype=np.float32)

        def convert_a(obs_dict):
            return np.concatenate([
                obs_dict['step_belief'].astype(np.float32),
                obs_dict['expected_tau'].astype(np.float32),
                obs_dict['memory_estimate_critical'].astype(np.float32),
                np.array([obs_dict['human_last_action']], dtype=np.float32),
            ])

        records = []
        cum_reward = 0.0

        while not ma_env.ma_state.is_done:
            h_obs = convert_h(h_obs_dict)
            a_obs = convert_a(a_obs_dict)
            h_action, _ = human_model.predict(h_obs, deterministic=deterministic)
            a_action, _ = assistant_model.predict(a_obs, deterministic=deterministic)

            h_obs_dict, a_obs_dict, reward, done, info = ma_env.step(
                int(h_action), int(a_action)
            )
            cum_reward += reward

            true_step = info['true_step']
            if true_step < ma_env.n_steps:
                true_id = int(ma_env.episode_order[true_step])
            else:
                true_id = ma_env.n_steps

            records.append(TickRecord(
                tick=len(records),
                true_step=true_step,
                true_identity=true_id,
                h_action=int(h_action),
                a_action=int(a_action),
                step_completed=info['step_completed'],
                failure=info['failure'],
                memory=info['memory'].copy(),
                obs_noise_state=info['obs_noise_state'],
                step_belief=ma_env.ma_state.step_belief.copy(),
                cumulative_reward=cum_reward,
                episode_order=ma_env.episode_order.copy(),
            ))

        return records, ma_env, params


def main():
    parser = argparse.ArgumentParser(
        description='Animate representative episodes from grid search')
    parser.add_argument('--results', default=str(
        PROJECT_ROOT / 'data' / 'results' /
        'grid_search_make_cereal_step_transition_durable.json'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fps', type=int, default=4)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--types', nargs='*', default=None,
                        help='Behavior types to animate (default: all found)')
    args = parser.parse_args()

    # Load grid results
    grid_data = json.load(open(args.results))
    task_name = grid_data['task']
    decay_regime = grid_data.get('decay_regime', 'step_transition')
    obs_regime = grid_data.get('obs_regime', 'durable')

    print(f"Task: {task_name}  decay={decay_regime}  obs={obs_regime}")
    print(f"\nFinding representatives:")
    reps = find_representatives(grid_data)

    types_to_animate = args.types or list(reps.keys())
    recorder = GridEpisodeRecorder()
    output_dir = PROJECT_ROOT / "results" / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    for btype in BEHAVIOR_ORDER:
        if btype not in types_to_animate or btype not in reps:
            continue

        rep = reps[btype]
        cf, cc = rep['c_fail'], rep['c_comm']
        label = BEHAVIOR_LABELS[btype]

        print(f"\n{'='*70}")
        print(f"Animating: {btype} — {label}")
        print(f"  cf={cf:.1f}  cc={cc:.4f}  "
              f"expected: narr={rep['narr']:.1f} q={rep['q']:.1f} "
              f"int={rep['interact']:.1f} fail={rep['fail']:.2f}")
        print(f"{'='*70}")

        try:
            records, ma_env, params = recorder.run_episode(
                task_name, cf, cc,
                decay_regime=decay_regime, obs_regime=obs_regime,
                seed=args.seed, deterministic=args.deterministic,
            )

            h_names, _ = ma_env.get_action_names()
            n_narr = sum(1 for r in records if h_names.get(r.h_action) == 'narrate')
            n_q = sum(1 for r in records if r.h_action in ma_env._question_id_to_step)
            n_fail = sum(1 for r in records if r.failure)

            print(f"  Episode: {len(records)} ticks, reward={records[-1].cumulative_reward:+.2f}, "
                  f"narr={n_narr}, q={n_q}, fail={n_fail}")

            # Use regime name in title
            regime_label = f"cf={cf:.1f} cc={cc:.2f}"

            animator = MAEpisodeAnimator(
                records, ma_env, params,
                task_name, regime_label, probe_id=None,
            )
            # Override title to show behavior type
            animator.probe_id = None

            output_path = output_dir / f"grid_phase_{btype}"
            animator.save(output_path, fps=args.fps)

        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone! Videos in: {output_dir}/")


if __name__ == '__main__':
    main()
