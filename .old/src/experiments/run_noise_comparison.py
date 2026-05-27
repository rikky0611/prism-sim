"""
obs_noise Comparison Experiment for MA-IPPO v2 (Belief State POMDP)

Trains and evaluates MA-IPPO under two observation noise levels:
  - noise=0.2: low noise (Bayesian updates are informative → narration may not emerge)
  - noise=0.5: high noise (Bayesian updates are less informative → narration should emerge)

For each noise condition × 3 cost regimes, runs:
  - Baseline evaluation (both-silent, always-narrate+silent)
  - MA-IPPO training (8 rounds × 20k steps)
  - Final evaluation (200 episodes)
  - Episode trace collection (5 representative episodes)

Output:
  data/results/ma_v2_noise02_make_cereal_all.json
  data/results/ma_v2_noise05_make_cereal_all.json
  models/ma_ippo_v2/<noise>/<regime>/

Usage:
  python3 src/experiments/run_noise_comparison.py
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training"))

from task_definitions import load_task_definitions, create_per_step_failure_costs
from ma_procedure_assistant_sim import (
    MAProcedureAssistantEnv,
    MASimulationParams,
    PassiveHumanPolicy,
    SilentAssistantPolicy,
    AlwaysNarrateHumanPolicy,
    RandomHumanPolicy,
    RandomAssistantPolicy,
)
from train_ma_ippo import (
    HumanGymWrapper,
    AssistantGymWrapper,
    train_ippo,
    evaluate_joint,
)


# ============================================================================
# NOISE CONDITIONS
# ============================================================================
NOISE_CONDITIONS = [
    {'label': 'noise02', 'obs_noise': 0.20, 'obs_noise_min': 0.02},
    {'label': 'noise05', 'obs_noise': 0.50, 'obs_noise_min': 0.05},
]

# ============================================================================
# COST REGIMES
# ============================================================================
REGIME_CONFIGS = [
    ('extremely_low',  dict(c_fail_scale=2.0,  c_int=1.0, c_nar=0.5)),
    ('balanced',       dict(c_fail_scale=15.0, c_int=1.0, c_nar=1.0)),
    ('extremely_high', dict(c_fail_scale=50.0, c_int=1.0, c_nar=2.0)),
]

N_ROUNDS = 8
STEPS_PER_ROUND = 20_000
N_EVAL = 200
N_TRACE_EPISODES = 5  # representative episodes to collect traces for


# ============================================================================
# EPISODE TRACE COLLECTION
# ============================================================================
def collect_episode_trace(
    human_model,
    asst_model,
    ma_env: MAProcedureAssistantEnv,
) -> List[Dict]:
    """Run one deterministic episode and record per-tick trace.

    Returns:
        List of dicts, one per tick:
            tick, true_step, h_action, a_action,
            step_belief_argmax, step_belief_entropy,
            memory_est_max (max over critical steps)
    """
    hw = HumanGymWrapper(ma_env, asst_model)
    aw = AssistantGymWrapper(ma_env, human_model)

    h_obs_dict, a_obs_dict = ma_env.reset()
    h_obs = hw._convert_human_obs(h_obs_dict)
    a_obs = aw._convert_assistant_obs(a_obs_dict)

    trace = []
    done = False
    tick = 0

    while not done and tick < 500:  # safety cap
        h_act, _ = human_model.predict(h_obs, deterministic=True)
        a_act, _ = asst_model.predict(a_obs, deterministic=True)

        belief = a_obs_dict['step_belief']
        # Entropy of belief distribution (nats)
        belief_safe = np.clip(belief, 1e-10, 1.0)
        entropy = float(-np.sum(belief_safe * np.log(belief_safe)))

        trace.append({
            'tick': tick,
            'true_step': int(ma_env.ma_state.current_step),
            'h_action': int(h_act),
            'a_action': int(a_act),
            'step_belief_argmax': int(np.argmax(belief)),
            'step_belief_entropy': round(entropy, 4),
            'memory_est_max': round(
                float(np.max(a_obs_dict['memory_estimate_critical']))
                if len(a_obs_dict['memory_estimate_critical']) > 0 else 0.0,
                4,
            ),
        })

        h_obs_dict, a_obs_dict, reward, done, info = ma_env.step(int(h_act), int(a_act))
        h_obs = hw._convert_human_obs(h_obs_dict)
        a_obs = aw._convert_assistant_obs(a_obs_dict)
        tick += 1

    return trace


# ============================================================================
# BASELINE EVALUATION
# ============================================================================
def eval_fixed_policies(h_policy, a_policy, ma_env, n=N_EVAL):
    """Evaluate a pair of fixed (non-RL) policies over n episodes."""
    rewards, failures, narrations, questions, ticks = [], [], [], [], []
    for _ in range(n):
        h_obs, a_obs = ma_env.reset()
        ep_r = 0.0
        done = False
        while not done:
            h_act = h_policy.get_action(h_obs)
            a_act = a_policy.get_action(a_obs)
            h_obs, a_obs, r, done, _ = ma_env.step(h_act, a_act)
            ep_r += r
        s = ma_env.ma_state
        rewards.append(ep_r)
        failures.append(s.total_failures)
        narrations.append(s.total_narrations)
        questions.append(s.total_questions)
        ticks.append(s.global_tick)
    return {
        'mean_reward': round(float(np.mean(rewards)), 4),
        'std_reward': round(float(np.std(rewards)), 4),
        'mean_failures': round(float(np.mean(failures)), 4),
        'mean_narrations': round(float(np.mean(narrations)), 4),
        'mean_questions': round(float(np.mean(questions)), 4),
        'mean_ticks': round(float(np.mean(ticks)), 4),
    }


# ============================================================================
# MAIN
# ============================================================================
def run_noise_condition(noise_cfg: Dict, task_def, results_dir: Path):
    """Run full 3-regime experiment for one noise condition."""
    label = noise_cfg['label']
    obs_noise = noise_cfg['obs_noise']
    obs_noise_min = noise_cfg['obs_noise_min']

    print(f'\n{"#"*70}', flush=True)
    print(f'# obs_noise = {obs_noise}  (label={label})', flush=True)
    print(f'{"#"*70}', flush=True)

    all_results: Dict[str, Any] = {
        'task': task_def.task_name,
        'obs_noise': obs_noise,
        'obs_noise_min': obs_noise_min,
        'regimes': {},
    }

    for regime_name, cfg in REGIME_CONFIGS:
        print(f'\n{"="*60}', flush=True)
        print(f'Regime: {regime_name}  (obs_noise={obs_noise})', flush=True)
        print(f'{"="*60}', flush=True)

        c_fail_per_step = create_per_step_failure_costs(task_def, cfg['c_fail_scale'])
        params = MASimulationParams(
            lambda_forget=0.03,
            delta_reminder=0.8,
            f0_base=0.6,
            k_memory=3.0,
            c_fail_per_step=c_fail_per_step,
            c_int=cfg['c_int'],
            c_nar=cfg['c_nar'],
            obs_noise=obs_noise,
            obs_noise_min=obs_noise_min,
            lambda_noise_recover=0.10,
        )

        ma_env = MAProcedureAssistantEnv(params, task_def)

        # --- Baselines ---
        print('Evaluating baselines...', flush=True)
        b_silent = eval_fixed_policies(
            PassiveHumanPolicy(), SilentAssistantPolicy(), ma_env
        )
        b_narrate = eval_fixed_policies(
            AlwaysNarrateHumanPolicy(), SilentAssistantPolicy(), ma_env
        )
        print(f'  Both silent:          reward={b_silent["mean_reward"]:.2f}, '
              f'failures={b_silent["mean_failures"]:.2f}', flush=True)
        print(f'  AlwaysNarrate+Silent: reward={b_narrate["mean_reward"]:.2f}, '
              f'failures={b_narrate["mean_failures"]:.2f}', flush=True)

        # --- MA-IPPO Training ---
        model_dir = (
            PROJECT_ROOT / 'models' / 'ma_ippo_v2' / label / task_def.task_name / regime_name
        )
        human_model, asst_model, training_log = train_ippo(
            params=params,
            task_def=task_def,
            n_rounds=N_ROUNDS,
            steps_per_round=STEPS_PER_ROUND,
            save_dir=model_dir,
            verbose=1,
        )

        # --- Final Evaluation ---
        final = evaluate_joint(human_model, asst_model, ma_env, n_episodes=N_EVAL)
        print(f'\nMA-IPPO final: reward={final["mean_reward"]:.2f}, '
              f'failures={final["mean_failures"]:.2f}, '
              f'narrations={final["mean_narrations"]:.2f}, '
              f'questions={final["mean_questions"]:.2f}', flush=True)

        # --- Episode Traces ---
        print(f'Collecting {N_TRACE_EPISODES} episode traces...', flush=True)
        traces = []
        for ep_i in range(N_TRACE_EPISODES):
            t = collect_episode_trace(human_model, asst_model, ma_env)
            traces.append(t)
            n_nar = sum(1 for r in t if r['h_action'] == 1)
            n_rem = sum(1 for r in t if r['a_action'] >= 2)
            print(f'  trace {ep_i}: len={len(t)}, narrations={n_nar}, reminds={n_rem}',
                  flush=True)

        all_results['regimes'][regime_name] = {
            'baselines': {
                'both_silent': b_silent,
                'always_narrate_silent': b_narrate,
            },
            'ma_ippo': {
                'training_log': training_log['rounds'],
                'final_eval': final,
                'episode_traces': traces,
            },
            'n_critical': ma_env.n_critical,
            'R_complete': ma_env.R_complete,
            'step_names': list(ma_env.procedural_steps),
            'critical_steps': ma_env.critical_steps,
            'n_human_actions': ma_env.n_human_actions,
            'n_assistant_actions': ma_env.n_assistant_actions,
        }

    # Save JSON
    out_path = results_dir / f'ma_v2_{label}_{task_def.task_name}_all.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResults saved → {out_path}', flush=True)
    return all_results


def main():
    tasks = load_task_definitions()
    task_def = tasks['make_cereal']

    results_dir = PROJECT_ROOT / 'data' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    for noise_cfg in NOISE_CONDITIONS:
        run_noise_condition(noise_cfg, task_def, results_dir)

    print('\n\nALL NOISE CONDITIONS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
