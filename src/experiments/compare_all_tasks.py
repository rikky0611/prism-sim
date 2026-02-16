"""
Compare RL and baseline policies across all 7 tasks.

Evaluates Random, Proactive, Reactive, and RL (PPO) policies
on each task and saves results for cross-task analysis.
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any
from datetime import datetime

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training"))

from task_definitions import load_task_definitions, TaskDefinition
from procedure_assistant_sim import (
    ProcedureAssistantEnv,
    SimulationParams,
    RandomAssistantPolicy,
    ProactiveReminderPolicy,
    ReactivePolicyHighCost,
)
from train_rl_policy import evaluate_policy

from stable_baselines3 import PPO


def evaluate_all_policies_on_task(
    task_name: str,
    task_def: TaskDefinition,
    params: SimulationParams,
    rl_model_path: str,
    n_episodes: int = 100
) -> Dict[str, Dict[str, Any]]:
    """Evaluate all policies on a single task.

    Args:
        task_name: Name of the task
        task_def: TaskDefinition for the task
        params: SimulationParams with memory/failure/cost parameters
        rl_model_path: Path to trained RL model
        n_episodes: Number of evaluation episodes

    Returns:
        Dictionary with results for each policy
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING: {task_name} ({task_def.n_steps} steps, {task_def.domain})")
    print(f"{'='*80}\n")

    # Apply task-specific costs
    params.apply_task_defaults(task_def)

    # Create environment
    env = ProcedureAssistantEnv(params, task_def)

    # Create baseline policies (adapted to task size)
    policies = {
        'Random': RandomAssistantPolicy(task_def.n_steps),
        'Proactive': ProactiveReminderPolicy(
            task_def.n_steps,
            memory_threshold=0.3,
            lookahead=1
        ),
        'Reactive': ReactivePolicyHighCost(
            task_def.n_steps,
            risk_threshold=0.25,
            params=params
        ),
    }

    results = {}

    # Evaluate baseline policies
    for policy_name, policy in policies.items():
        print(f"  Evaluating {policy_name}...")
        results[policy_name] = evaluate_policy(env, policy, task_def, n_episodes, seed=42)
        print(f"    Reward: {results[policy_name]['mean_reward']:.2f} ± "
              f"{results[policy_name]['std_reward']:.2f}")
        print(f"    Interruptions: {results[policy_name]['mean_interruptions']:.2f}, "
              f"Failures: {results[policy_name]['mean_failures']:.2f}")

    # Load and evaluate RL model
    print(f"  Evaluating RL (PPO)...")
    try:
        rl_model = PPO.load(rl_model_path)
        results['RL_PPO'] = evaluate_policy(env, rl_model, task_def, n_episodes, seed=42)
        print(f"    Reward: {results['RL_PPO']['mean_reward']:.2f} ± "
              f"{results['RL_PPO']['std_reward']:.2f}")
        print(f"    Interruptions: {results['RL_PPO']['mean_interruptions']:.2f}, "
              f"Failures: {results['RL_PPO']['mean_failures']:.2f}")
    except Exception as e:
        print(f"    ✗ Failed to load RL model: {e}")
        results['RL_PPO'] = {
            'error': str(e),
            'mean_reward': float('-inf'),
            'std_reward': 0.0,
            'mean_interruptions': 0.0,
            'mean_failures': 0.0,
        }

    return results


def compare_all_tasks(
    cost_regime: str = "balanced",
    n_episodes: int = 100,
    lambda_forget: float = 0.05,
    f0_base: float = 0.3,
    k_memory: float = 2.0,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Compare all policies across all tasks.

    Args:
        cost_regime: Cost regime identifier
        n_episodes: Number of evaluation episodes per policy
        lambda_forget: Forgetting rate parameter
        f0_base: Base failure probability
        k_memory: Memory effect on failure

    Returns:
        Nested dictionary: tasks -> policies -> metrics
    """
    print("#" * 80)
    print("# CROSS-TASK POLICY COMPARISON")
    print("#" * 80)
    print(f"\nEvaluation parameters:")
    print(f"  Episodes per policy: {n_episodes}")
    print(f"  Cost regime: {cost_regime}")
    print(f"  Lambda (forget): {lambda_forget}")
    print(f"  f0 (base fail): {f0_base}")
    print(f"  k (memory effect): {k_memory}")
    print()

    # Load all task definitions
    tasks = load_task_definitions()
    print(f"Loaded {len(tasks)} tasks\n")

    # Create base params
    base_params = SimulationParams(
        lambda_forget=lambda_forget,
        f0_base=f0_base,
        k_memory=k_memory,
        c_int=5.0,
        c_fail_base=20.0,
    )

    # Results storage
    all_results = {}
    start_time = datetime.now()

    # Evaluate each task
    for i, (task_name, task_def) in enumerate(tasks.items(), 1):
        print(f"\n[{i}/{len(tasks)}] Processing {task_name}...")

        # Determine RL model path
        rl_model_path = (
            PROJECT_ROOT / "models" / f"ppo_{task_name}_{cost_regime}" /
            f"ppo_{task_name}_{cost_regime}" / "final_model"
        )

        if not rl_model_path.exists():
            # Try alternative path structure
            rl_model_path = (
                PROJECT_ROOT / "models" / f"ppo_{task_name}_{cost_regime}" /
                "final_model"
            )

        if not rl_model_path.exists():
            print(f"  ⚠️  RL model not found at {rl_model_path}")
            print(f"     Skipping RL evaluation for {task_name}")

        # Evaluate all policies on this task
        task_results = evaluate_all_policies_on_task(
            task_name=task_name,
            task_def=task_def,
            params=base_params,
            rl_model_path=str(rl_model_path),
            n_episodes=n_episodes
        )

        all_results[task_name] = task_results

    # Final summary
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()

    print("\n" + "#" * 80)
    print("# EVALUATION COMPLETE")
    print("#" * 80)
    print(f"\nTotal evaluation time: {total_duration/60:.1f} minutes")
    print()

    # Save results
    results_path = PROJECT_ROOT / "data" / "results" / f"cross_task_comparison_{cost_regime}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    results_serializable = {}
    for task_name, task_results in all_results.items():
        results_serializable[task_name] = {}
        for policy_name, metrics in task_results.items():
            results_serializable[task_name][policy_name] = {
                'mean_reward': float(metrics['mean_reward']),
                'std_reward': float(metrics['std_reward']),
                'mean_interruptions': float(metrics['mean_interruptions']),
                'std_interruptions': float(metrics.get('std_interruptions', 0.0)),
                'mean_failures': float(metrics['mean_failures']),
                'std_failures': float(metrics.get('std_failures', 0.0)),
                'error': metrics.get('error', None),
            }

    # Add metadata
    task_metadata = {}
    for task_name, task_def in tasks.items():
        task_metadata[task_name] = {
            'n_steps': task_def.n_steps,
            'domain': task_def.domain,
            'base_failure_cost': task_def.base_failure_cost,
            'interruption_cost': task_def.interruption_cost,
        }

    output = {
        'metadata': {
            'cost_regime': cost_regime,
            'n_episodes': n_episodes,
            'evaluated_at': start_time.isoformat(),
            'duration_seconds': total_duration,
        },
        'params': {
            'lambda_forget': lambda_forget,
            'f0_base': f0_base,
            'k_memory': k_memory,
        },
        'tasks': task_metadata,
        'results': results_serializable,
    }

    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {results_path}")
    print()

    # Print summary table
    print_summary_table(all_results, tasks)

    return all_results


def print_summary_table(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    tasks: Dict[str, TaskDefinition]
):
    """Print summary table of results."""
    print("\n" + "="*80)
    print("SUMMARY: Mean Rewards by Task and Policy")
    print("="*80)

    # Header
    policies = ['Random', 'Proactive', 'Reactive', 'RL_PPO']
    print(f"{'Task':<20} {'Steps':<8} {'Domain':<12} ", end='')
    for policy in policies:
        print(f"{policy:<12} ", end='')
    print()
    print("-" * 80)

    # Rows
    for task_name in sorted(tasks.keys(), key=lambda t: tasks[t].n_steps):
        task_def = tasks[task_name]
        task_results = results[task_name]

        print(f"{task_name:<20} {task_def.n_steps:<8} {task_def.domain:<12} ", end='')

        for policy in policies:
            if policy in task_results and 'error' not in task_results[policy]:
                reward = task_results[policy]['mean_reward']
                print(f"{reward:<12.1f} ", end='')
            else:
                print(f"{'N/A':<12} ", end='')
        print()

    print("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compare policies across all procedural tasks'
    )
    parser.add_argument(
        '--regime',
        type=str,
        default='balanced',
        help='Cost regime name (default: balanced)'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Evaluation episodes per policy (default: 100)'
    )
    parser.add_argument(
        '--lambda-forget',
        type=float,
        default=0.05,
        help='Forgetting rate (default: 0.05)'
    )
    parser.add_argument(
        '--f0-base',
        type=float,
        default=0.3,
        help='Base failure probability (default: 0.3)'
    )
    parser.add_argument(
        '--k-memory',
        type=float,
        default=2.0,
        help='Memory effect on failure (default: 2.0)'
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(42)

    # Run comparison
    results = compare_all_tasks(
        cost_regime=args.regime,
        n_episodes=args.episodes,
        lambda_forget=args.lambda_forget,
        f0_base=args.f0_base,
        k_memory=args.k_memory,
    )

    print("\n" + "="*80)
    print(f"Ready for visualization with visualize_cross_task.py")
    print("="*80)


if __name__ == "__main__":
    main()
