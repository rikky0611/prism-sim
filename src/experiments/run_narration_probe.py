"""
Regime-based MA-IPPO Experiment Runner

Runs MA-IPPO training across failure cost regimes with specified
communication cost and memory decay regimes.

Three orthogonal axes:
  - Failure cost regime: extremely_low / balanced / extremely_high
  - Communication cost regime: default / cheap_narrate / ...
  - Memory decay regime: default / step_transition / ...

Usage:
    # Run all failure regimes with cheap_narrate comm + step_transition decay
    python run_narration_probe.py --task make_tea \
        --comm-regime cheap_narrate --decay-regime step_transition --rounds 15

    # Run across all tasks
    python run_narration_probe.py --task all \
        --comm-regime cheap_narrate --decay-regime step_transition --rounds 15

Output:
    data/results/ma_<task>_<comm>_<decay>.json
    models/ma_ippo/<task>/<fail_regime>/
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training"))

from task_definitions import load_task_definitions
from regime_definitions import (
    build_params,
    COMM_COST_REGIMES,
    MEMORY_DECAY_REGIMES,
    FAILURE_COST_SCALES,
    OBS_NOISE_REGIMES,
)
from run_ma_experiments import run_experiment_for_regime


def run_regime_experiment(
    task_name: str,
    comm_regime: str = 'default',
    decay_regime: str = 'default',
    obs_regime: str = 'default',
    fail_regimes: list = None,
    n_rounds: int = 15,
    steps_per_round: int = 50_000,
    n_eval_episodes: int = 200,
) -> Dict[str, Any]:
    """Run experiments for specified regime combination.

    Args:
        task_name: Task to train on
        comm_regime: Communication cost regime name
        decay_regime: Memory decay regime name
        obs_regime: Observation noise regime name
        fail_regimes: List of failure regimes (default: all)
        n_rounds: IPPO training rounds
        steps_per_round: Env steps per agent per round
        n_eval_episodes: Episodes for evaluation

    Returns:
        Results dict with per-regime metrics
    """
    if fail_regimes is None:
        fail_regimes = list(FAILURE_COST_SCALES.keys())

    tasks = load_task_definitions()
    task_def = tasks[task_name]

    comm = COMM_COST_REGIMES[comm_regime]
    decay = MEMORY_DECAY_REGIMES[decay_regime]
    obs = OBS_NOISE_REGIMES[obs_regime]

    import math
    half_life = math.log(2) / obs.lambda_noise_recover

    print(f"\n{'='*70}")
    print(f"Task: {task_name}  comm={comm_regime}  decay={decay_regime}  obs={obs_regime}")
    print(f"  c_nar={comm.c_nar}, c_remind={comm.c_remind}, "
          f"c_q={comm.c_q}, c_confirm={comm.c_confirm}")
    print(f"  lambda_forget={decay.lambda_forget}, memory_init={decay.memory_init}")
    print(f"  obs_noise={obs.obs_noise}, obs_noise_min={obs.obs_noise_min}, "
          f"lambda_noise_recover={obs.lambda_noise_recover} (half-life={half_life:.1f} ticks)")
    print(f"{'='*70}")

    all_results: Dict[str, Any] = {
        'task': task_name,
        'comm_regime': comm_regime,
        'decay_regime': decay_regime,
        'obs_regime': obs_regime,
        'param_overrides': {
            'c_nar': comm.c_nar, 'c_remind': comm.c_remind,
            'c_q': comm.c_q, 'c_confirm': comm.c_confirm,
            'lambda_forget': decay.lambda_forget,
            'memory_init': decay.memory_init,
            'lambda_noise_recover': obs.lambda_noise_recover,
        },
        'regimes': {},
    }

    for fail_regime in fail_regimes:
        params = build_params(task_def, fail_regime, comm_regime, decay_regime, obs_regime)
        result = run_experiment_for_regime(
            task_name=task_name,
            regime_name=fail_regime,
            params=params,
            task_def=task_def,
            n_rounds=n_rounds,
            steps_per_round=steps_per_round,
            n_eval_episodes=n_eval_episodes,
        )
        all_results['regimes'][fail_regime] = result

    return all_results


def print_summary(results: Dict[str, Any]):
    """Print narration/communication summary."""
    task = results['task']
    comm = results['comm_regime']
    decay = results['decay_regime']

    print(f"\nSummary for {task} (comm={comm}, decay={decay}):")
    for regime, res in results['regimes'].items():
        b = res['baselines']
        ippo = res['ma_ippo']['final_eval']
        print(f"  {regime:18s}: "
              f"reward={ippo['mean_reward']:+.2f}, "
              f"narr={ippo['mean_narrations']:.2f}, "
              f"quest={ippo['mean_questions']:.2f}, "
              f"interact={ippo['mean_interactions']:.2f}, "
              f"fail={ippo['mean_failures']:.2f}")


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Run MA-IPPO experiments with regime-based configuration'
    )
    parser.add_argument('--task', default='make_tea',
                        help='Task name or "all" (default: make_tea)')
    parser.add_argument('--comm-regime', default='cheap_narrate',
                        choices=list(COMM_COST_REGIMES.keys()),
                        help='Communication cost regime (default: cheap_narrate)')
    parser.add_argument('--decay-regime', default='step_transition',
                        choices=list(MEMORY_DECAY_REGIMES.keys()),
                        help='Memory decay regime (default: step_transition)')
    parser.add_argument('--obs-regime', default='default',
                        choices=list(OBS_NOISE_REGIMES.keys()),
                        help='Observation noise regime (default: default)')
    parser.add_argument('--fail-regime', default=None,
                        choices=list(FAILURE_COST_SCALES.keys()),
                        help='Specific failure regime (default: all)')
    parser.add_argument('--rounds', type=int, default=15,
                        help='Training rounds (default: 15)')
    parser.add_argument('--steps', type=int, default=50_000,
                        help='Steps per agent per round (default: 50000)')
    parser.add_argument('--eval-episodes', type=int, default=200,
                        help='Evaluation episodes (default: 200)')
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    fail_regimes = [args.fail_regime] if args.fail_regime else None

    # Resolve task list
    if args.task == 'all':
        task_list = list(load_task_definitions().keys())
    else:
        task_list = [args.task]

    for task_name in task_list:
        results = run_regime_experiment(
            task_name=task_name,
            comm_regime=args.comm_regime,
            decay_regime=args.decay_regime,
            obs_regime=args.obs_regime,
            fail_regimes=fail_regimes,
            n_rounds=args.rounds,
            steps_per_round=args.steps,
            n_eval_episodes=args.eval_episodes,
        )

        out_path = (results_dir /
                    f"ma_{task_name}_{args.comm_regime}_{args.decay_regime}_{args.obs_regime}.json")
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")

        print_summary(results)


if __name__ == '__main__':
    main()
