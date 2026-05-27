"""
Multi-Agent IPPO Experiment Runner

Trains and evaluates the MA-IPPO setup across cost regimes and tasks.
Compares against baselines:
  - PassiveHuman + RL Assistant (SA-like)
  - RL Human + SilentAssistant
  - Both agents random
  - MA-IPPO (both RL)

Usage:
    cd src/experiments
    python run_ma_experiments.py --task make_cereal --rounds 15 --steps 100000

Output:
    data/results/ma_experiments_<task>.json
    models/ma_ippo/<task>/<regime>/
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training"))

from task_definitions import load_task_definitions, create_per_step_failure_costs
from ma_procedure_assistant_sim import (
    MAProcedureAssistantEnv,
    MASimulationParams,
    RandomHumanPolicy,
    RandomAssistantPolicy,
    PassiveHumanPolicy,
    SilentAssistantPolicy,
    AlwaysNarrateHumanPolicy,
    HeuristicReminderAssistantPolicy,
)
from train_ma_ippo import (
    HumanGymWrapper,
    AssistantGymWrapper,
    train_ippo,
    define_ma_cost_regimes,
    evaluate_joint,
)

try:
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


# ============================================================================
# BASELINE EVALUATION
# ============================================================================
def evaluate_baseline_joint(
    human_policy,
    assistant_policy,
    ma_env: MAProcedureAssistantEnv,
    n_episodes: int = 200,
) -> Dict[str, float]:
    """Evaluate a pair of fixed (non-RL) policies."""
    human_wrapper = HumanGymWrapper(ma_env, assistant_policy)

    total_rewards = []
    total_failures = []
    total_narrations = []
    total_questions = []
    total_reminds = []
    total_confirms = []
    total_ticks = []
    total_tracking_acc = []

    for _ in range(n_episodes):
        # Reset policies that track per-episode state
        if hasattr(human_policy, 'reset'):
            human_policy.reset()
        if hasattr(assistant_policy, 'reset'):
            assistant_policy.reset()

        h_obs_dict, a_obs_dict = ma_env.reset()
        ep_reward = 0.0
        ep_tracking = []
        done = False

        while not done:
            h_obs = human_wrapper._convert_human_obs(h_obs_dict)
            a_obs = human_wrapper._convert_assistant_obs(a_obs_dict)

            h_act = human_policy.get_action(h_obs_dict)
            a_act = assistant_policy.get_action(a_obs_dict)

            h_obs_dict, a_obs_dict, reward, done, info = ma_env.step(h_act, a_act)
            ep_reward += reward
            mc = info.get('tracking_map_correct')
            if mc is not None:
                ep_tracking.append(mc)

        state = ma_env.ma_state
        total_rewards.append(ep_reward)
        total_failures.append(state.total_failures)
        total_narrations.append(state.total_narrations)
        total_questions.append(state.total_questions)
        total_reminds.append(state.total_reminds)
        total_confirms.append(state.total_confirms)
        total_ticks.append(state.global_tick)
        total_tracking_acc.append(float(np.mean(ep_tracking)) if ep_tracking else np.nan)

    return {
        'mean_reward': float(np.mean(total_rewards)),
        'std_reward': float(np.std(total_rewards)),
        'mean_failures': float(np.mean(total_failures)),
        'mean_narrations': float(np.mean(total_narrations)),
        'mean_questions': float(np.mean(total_questions)),
        'mean_reminds': float(np.mean(total_reminds)),
        'mean_confirms': float(np.mean(total_confirms)),
        'mean_ticks': float(np.mean(total_ticks)),
        'mean_tracking_map_acc': float(np.nanmean(total_tracking_acc)),
        'std_tracking_map_acc': float(np.nanstd(total_tracking_acc)),
    }


def evaluate_rl_vs_fixed(
    rl_model,
    fixed_policy,
    ma_env: MAProcedureAssistantEnv,
    rl_is_human: bool,
    n_episodes: int = 200,
) -> Dict[str, float]:
    """Evaluate RL agent paired with a fixed policy."""
    total_rewards, total_failures, total_narrations, total_questions, total_ticks = [], [], [], [], []
    total_tracking_acc = []
    if rl_is_human:
        human_wrapper = HumanGymWrapper(ma_env, fixed_policy)
        for _ in range(n_episodes):
            h_obs_dict, a_obs_dict = ma_env.reset()
            h_obs = human_wrapper._convert_human_obs(h_obs_dict)
            ep_reward = 0.0
            ep_tracking = []
            done = False
            while not done:
                a_act = fixed_policy.get_action(a_obs_dict)
                h_act, _ = rl_model.predict(h_obs, deterministic=True)
                h_obs_dict, a_obs_dict, reward, done, info = ma_env.step(int(h_act), int(a_act))
                h_obs = human_wrapper._convert_human_obs(h_obs_dict)
                ep_reward += reward
                mc = info.get('tracking_map_correct')
                if mc is not None:
                    ep_tracking.append(mc)
            state = ma_env.ma_state
            total_rewards.append(ep_reward)
            total_failures.append(state.total_failures)
            total_narrations.append(state.total_narrations)
            total_questions.append(state.total_questions)
            total_ticks.append(state.global_tick)
            total_tracking_acc.append(float(np.mean(ep_tracking)) if ep_tracking else np.nan)
    else:
        assistant_wrapper = AssistantGymWrapper(ma_env, fixed_policy)
        for _ in range(n_episodes):
            h_obs_dict, a_obs_dict = ma_env.reset()
            a_obs = assistant_wrapper._convert_assistant_obs(a_obs_dict)
            ep_reward = 0.0
            ep_tracking = []
            done = False
            while not done:
                h_act = fixed_policy.get_action(h_obs_dict)
                a_act, _ = rl_model.predict(a_obs, deterministic=True)
                h_obs_dict, a_obs_dict, reward, done, info = ma_env.step(int(h_act), int(a_act))
                a_obs = assistant_wrapper._convert_assistant_obs(a_obs_dict)
                ep_reward += reward
                mc = info.get('tracking_map_correct')
                if mc is not None:
                    ep_tracking.append(mc)
            state = ma_env.ma_state
            total_rewards.append(ep_reward)
            total_failures.append(state.total_failures)
            total_narrations.append(state.total_narrations)
            total_questions.append(state.total_questions)
            total_ticks.append(state.global_tick)
            total_tracking_acc.append(float(np.mean(ep_tracking)) if ep_tracking else np.nan)

    return {
        'mean_reward': float(np.mean(total_rewards)),
        'std_reward': float(np.std(total_rewards)),
        'mean_failures': float(np.mean(total_failures)),
        'mean_narrations': float(np.mean(total_narrations)),
        'mean_questions': float(np.mean(total_questions)),
        'mean_ticks': float(np.mean(total_ticks)),
        'mean_tracking_map_acc': float(np.nanmean(total_tracking_acc)),
        'std_tracking_map_acc': float(np.nanstd(total_tracking_acc)),
    }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
def run_experiment_for_regime(
    task_name: str,
    regime_name: str,
    params: MASimulationParams,
    task_def,
    n_rounds: int,
    steps_per_round: int,
    n_eval_episodes: int,
    min_rounds: int = 8,
    patience: int = 6,
    eval_episodes: int = 100,
) -> Dict[str, Any]:
    """Run full experiment for one task-regime combination.

    n_rounds is the MAX rounds (early stopping may stop sooner). The reported
    final_eval is the BEST checkpoint evaluated over n_eval_episodes episodes
    (use a large value, e.g. 300, to reduce evaluation variance). eval_episodes
    is the per-round eval count used for best-checkpoint selection / early stop.
    """
    print(f"\n{'='*60}")
    print(f"Task: {task_name}  Regime: {regime_name}")
    print(f"{'='*60}")

    ma_env = MAProcedureAssistantEnv(params, task_def)
    n_h = ma_env.n_human_actions
    n_a = ma_env.n_assistant_actions

    result: Dict[str, Any] = {
        'task': task_name,
        'regime': regime_name,
        'n_critical': ma_env.n_critical,
        'R_complete': ma_env.R_complete,
        'baselines': {},
        'ma_ippo': {},
    }

    # --- Baselines (no RL) ---
    print("\nEvaluating baselines...")

    # 1. Both silent
    b1 = evaluate_baseline_joint(
        PassiveHumanPolicy(), SilentAssistantPolicy(), ma_env, n_eval_episodes
    )
    result['baselines']['both_silent'] = b1
    print(f"  Both silent:         reward={b1['mean_reward']:.2f}, failures={b1['mean_failures']:.2f}")

    # 2. Human always narrates + silent assistant
    b2 = evaluate_baseline_joint(
        AlwaysNarrateHumanPolicy(), SilentAssistantPolicy(), ma_env, n_eval_episodes
    )
    result['baselines']['always_narrate_silent_asst'] = b2
    print(f"  AlwaysNarrate+Silent: reward={b2['mean_reward']:.2f}, failures={b2['mean_failures']:.2f}")

    # 3. Both random
    b3 = evaluate_baseline_joint(
        RandomHumanPolicy(n_h), RandomAssistantPolicy(n_a), ma_env, n_eval_episodes
    )
    result['baselines']['both_random'] = b3
    print(f"  Both random:         reward={b3['mean_reward']:.2f}, failures={b3['mean_failures']:.2f}")

    # --- MA-IPPO Training ---
    model_dir = PROJECT_ROOT / "models" / "ma_ippo" / task_name / regime_name
    human_model, assistant_model, training_log = train_ippo(
        params=params,
        task_def=task_def,
        n_rounds=n_rounds,
        steps_per_round=steps_per_round,
        save_dir=model_dir,
        verbose=1,
        min_rounds=min_rounds,
        patience=patience,
        eval_episodes=eval_episodes,
    )

    result['ma_ippo']['training_log'] = training_log['rounds']
    result['ma_ippo']['best_round'] = training_log.get('best_round')
    result['ma_ippo']['stopped_round'] = training_log.get('stopped_round')

    # Final evaluation on the BEST checkpoint (train_ippo returns best models),
    # over n_eval_episodes (large -> low variance) for the reported number.
    final_metrics = evaluate_joint(
        human_model, assistant_model, ma_env, n_episodes=n_eval_episodes
    )
    result['ma_ippo']['final_eval'] = final_metrics
    print(f"\nMA-IPPO best-checkpoint eval (round {training_log.get('best_round')}, "
          f"{n_eval_episodes} eps): reward={final_metrics['mean_reward']:.2f}, "
          f"failures={final_metrics['mean_failures']:.2f}")

    return result


def run_all_experiments(
    task_name: str,
    comm_regime: str = 'default',
    decay_regime: str = 'default',
    obs_regime: str = 'default',
    n_rounds: int = 15,
    steps_per_round: int = 50_000,
    n_eval_episodes: int = 200,
) -> Dict[str, Any]:
    """Run experiments across all failure regimes for a given task."""
    tasks = load_task_definitions()
    if task_name not in tasks:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(tasks.keys())}")
    task_def = tasks[task_name]

    regimes = define_ma_cost_regimes(task_def, comm_regime, decay_regime, obs_regime)
    all_results = {
        'task': task_name,
        'comm_regime': comm_regime,
        'decay_regime': decay_regime,
        'obs_regime': obs_regime,
        'regimes': {},
    }

    for regime_name, params in regimes.items():
        regime_result = run_experiment_for_regime(
            task_name=task_name,
            regime_name=regime_name,
            params=params,
            task_def=task_def,
            n_rounds=n_rounds,
            steps_per_round=steps_per_round,
            n_eval_episodes=n_eval_episodes,
        )
        all_results['regimes'][regime_name] = regime_result

    return all_results


# ============================================================================
# CLI
# ============================================================================
def main():
    from regime_definitions import (
        COMM_COST_REGIMES, MEMORY_DECAY_REGIMES, FAILURE_COST_SCALES,
        OBS_NOISE_REGIMES, build_params,
    )

    parser = argparse.ArgumentParser(description='Run MA-IPPO experiments')
    parser.add_argument('--task', default='make_cereal',
                        help='Task name (default: make_cereal)')
    parser.add_argument('--fail-regime', default=None,
                        choices=list(FAILURE_COST_SCALES.keys()),
                        help='Failure cost regime (default: all)')
    parser.add_argument('--comm-regime', default='default',
                        choices=list(COMM_COST_REGIMES.keys()),
                        help='Communication cost regime (default: default)')
    parser.add_argument('--decay-regime', default='default',
                        choices=list(MEMORY_DECAY_REGIMES.keys()),
                        help='Memory decay regime (default: default)')
    parser.add_argument('--obs-regime', default='default',
                        choices=list(OBS_NOISE_REGIMES.keys()),
                        help='Observation noise regime (default: default)')
    parser.add_argument('--rounds', type=int, default=15,
                        help='Training rounds per regime (default: 15)')
    parser.add_argument('--steps', type=int, default=50_000,
                        help='Steps per agent per round (default: 50000)')
    parser.add_argument('--eval-episodes', type=int, default=200,
                        help='Evaluation episodes (default: 200)')
    args = parser.parse_args()

    tasks = load_task_definitions()
    if args.task not in tasks:
        print(f"Unknown task '{args.task}'. Available: {list(tasks.keys())}")
        sys.exit(1)
    task_def = tasks[args.task]

    results_dir = PROJECT_ROOT / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"{args.comm_regime}_{args.decay_regime}_{args.obs_regime}"

    if args.fail_regime is not None:
        # Single failure regime
        params = build_params(task_def, args.fail_regime,
                              args.comm_regime, args.decay_regime, args.obs_regime)
        result = run_experiment_for_regime(
            task_name=args.task,
            regime_name=args.fail_regime,
            params=params,
            task_def=task_def,
            n_rounds=args.rounds,
            steps_per_round=args.steps,
            n_eval_episodes=args.eval_episodes,
        )
        out_path = results_dir / f"ma_{args.task}_{args.fail_regime}_{suffix}.json"
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {out_path}")
    else:
        # All failure regimes
        results = run_all_experiments(
            task_name=args.task,
            comm_regime=args.comm_regime,
            decay_regime=args.decay_regime,
            obs_regime=args.obs_regime,
            n_rounds=args.rounds,
            steps_per_round=args.steps,
            n_eval_episodes=args.eval_episodes,
        )
        out_path = results_dir / f"ma_{args.task}_all_{suffix}.json"
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nAll results saved to {out_path}")


if __name__ == '__main__':
    main()
