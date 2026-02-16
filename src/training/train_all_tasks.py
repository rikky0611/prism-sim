"""
Train RL policies for all 7 procedural tasks.

This script trains PPO models for each task independently,
saving models and summary results for cross-task analysis.
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
from procedure_assistant_sim import SimulationParams
from train_rl_policy import train_ppo_policy


def train_all_tasks(
    timesteps_per_task: int = 100000,
    cost_regime: str = "balanced",
    lambda_forget: float = 0.05,
    f0_base: float = 0.3,
    k_memory: float = 2.0,
) -> Dict[str, Dict[str, Any]]:
    """Train PPO models for all 7 tasks.

    Args:
        timesteps_per_task: Number of training timesteps per task
        cost_regime: Cost regime identifier (for naming)
        lambda_forget: Forgetting rate parameter
        f0_base: Base failure probability
        k_memory: Memory effect on failure

    Returns:
        Dictionary mapping task names to training summaries
    """
    print("#" * 80)
    print("# MULTI-TASK RL TRAINING")
    print("#" * 80)
    print(f"\nTraining parameters:")
    print(f"  Timesteps per task: {timesteps_per_task:,}")
    print(f"  Cost regime: {cost_regime}")
    print(f"  Lambda (forget): {lambda_forget}")
    print(f"  f0 (base fail): {f0_base}")
    print(f"  k (memory effect): {k_memory}")
    print()

    # Load all task definitions
    tasks = load_task_definitions()
    print(f"Loaded {len(tasks)} tasks:")
    for task_name, task_def in tasks.items():
        print(f"  - {task_name}: {task_def.n_steps} steps ({task_def.domain})")
    print()

    # Create base params (will be overridden by task-specific costs)
    base_params = SimulationParams(
        lambda_forget=lambda_forget,
        f0_base=f0_base,
        k_memory=k_memory,
        c_int=5.0,  # Will be overridden
        c_fail_base=20.0,  # Will be overridden
    )

    # Training summary
    training_summary = {}
    start_time = datetime.now()

    # Train each task
    for i, (task_name, task_def) in enumerate(tasks.items(), 1):
        print("=" * 80)
        print(f"TASK {i}/{len(tasks)}: {task_name}")
        print("=" * 80)

        task_start = datetime.now()

        # Copy params and apply task defaults
        task_params = SimulationParams(
            lambda_forget=base_params.lambda_forget,
            f0_base=base_params.f0_base,
            k_memory=base_params.k_memory,
            c_int=5.0,
            c_fail_base=20.0,
        )
        task_params.apply_task_defaults(task_def)

        print(f"\nTask properties:")
        print(f"  Steps: {task_def.n_steps}")
        print(f"  Domain: {task_def.domain}")
        print(f"  Base failure cost: {task_def.base_failure_cost}")
        print(f"  Interruption cost: {task_def.interruption_cost}")
        print(f"  Cost ratio (fail/int): {task_def.base_failure_cost / task_def.interruption_cost:.2f}")
        print()

        # Train model
        save_path = f"ppo_{task_name}_{cost_regime}"
        try:
            model = train_ppo_policy(
                params=task_params,
                task_def=task_def,
                total_timesteps=timesteps_per_task,
                save_path=save_path
            )

            task_end = datetime.now()
            task_duration = (task_end - task_start).total_seconds()

            training_summary[task_name] = {
                'success': True,
                'n_steps': task_def.n_steps,
                'domain': task_def.domain,
                'timesteps': timesteps_per_task,
                'duration_seconds': task_duration,
                'save_path': save_path,
                'params': {
                    'c_int': task_params.c_int,
                    'c_fail_base': task_params.c_fail_base,
                    'lambda_forget': task_params.lambda_forget,
                    'f0_base': task_params.f0_base,
                    'k_memory': task_params.k_memory,
                }
            }

            print(f"\n✓ {task_name} training completed in {task_duration:.1f}s")

        except Exception as e:
            print(f"\n✗ {task_name} training FAILED: {e}")
            import traceback
            traceback.print_exc()

            training_summary[task_name] = {
                'success': False,
                'error': str(e),
                'n_steps': task_def.n_steps,
                'domain': task_def.domain,
            }

        print()

    # Final summary
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()

    print("#" * 80)
    print("# TRAINING COMPLETE")
    print("#" * 80)
    print(f"\nTotal training time: {total_duration/60:.1f} minutes")
    print(f"Successful: {sum(1 for s in training_summary.values() if s['success'])}/{len(tasks)}")
    print()

    # Save summary
    summary_path = PROJECT_ROOT / "data" / "results" / f"all_tasks_{cost_regime}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, 'w') as f:
        json.dump({
            'metadata': {
                'cost_regime': cost_regime,
                'timesteps_per_task': timesteps_per_task,
                'total_duration_seconds': total_duration,
                'trained_at': start_time.isoformat(),
            },
            'base_params': {
                'lambda_forget': lambda_forget,
                'f0_base': f0_base,
                'k_memory': k_memory,
            },
            'tasks': training_summary
        }, f, indent=2)

    print(f"Training summary saved to: {summary_path}")
    print()

    return training_summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train RL policies for all procedural tasks'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=100000,
        help='Training timesteps per task (default: 100000)'
    )
    parser.add_argument(
        '--regime',
        type=str,
        default='balanced',
        help='Cost regime name (default: balanced)'
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

    # Train all tasks
    summary = train_all_tasks(
        timesteps_per_task=args.timesteps,
        cost_regime=args.regime,
        lambda_forget=args.lambda_forget,
        f0_base=args.f0_base,
        k_memory=args.k_memory,
    )

    # Print final status
    successful = [name for name, info in summary.items() if info['success']]
    failed = [name for name, info in summary.items() if not info['success']]

    if successful:
        print("✓ Successfully trained:")
        for task_name in successful:
            print(f"  - {task_name}")
        print()

    if failed:
        print("✗ Failed to train:")
        for task_name in failed:
            print(f"  - {task_name}: {summary[task_name].get('error', 'Unknown error')}")
        print()

    print("="*80)
    print(f"Ready for cross-task evaluation with compare_all_tasks.py")
    print("="*80)


if __name__ == "__main__":
    main()
