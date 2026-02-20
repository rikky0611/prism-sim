"""
Cross-Task Multi-Regime Training - OVERNIGHT EXPERIMENT

Train RL policies for ALL 7 tasks across 3 cost regimes.
Total: 21 models (7 tasks × 3 regimes)

Key Changes from Previous Experiments:
1. Sparse action space: Only 2-4 critical steps per task (conservative selection)
2. Higher baseline failure rate: f0_base=0.6 (60% → makes reminders 2× more beneficial)
3. Disabled observation noise: Agent sees true step (no confusion)
4. Off-timing penalty: Penalize reminding irrelevant steps

Expected Runtime: ~17 min/model × 21 models ≈ 6-7 hours

Usage:
    python train_cross_task_all_regimes.py --timesteps 200000 --seed 42
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import sys
import numpy as np
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training"))

from procedure_assistant_sim import SimulationParams
from task_definitions import TaskDefinition, get_task_definition, create_per_step_failure_costs
from train_rl_policy import train_ppo_policy


def get_all_tasks() -> List[str]:
    """Return all 7 task names."""
    return [
        'make_cereal',      # 8 steps, 2 critical (pour_cereal, pour_milk)
        'make_coffee',      # 8 steps, 1 critical (brew_coffee)
        'make_tea',         # 9 steps, 2 critical (heat_water, pour_water)
        'make_sandwich',    # 9 steps, 1 critical (prepare_sandwich)
        'make_stencil',     # 17 steps, 4 critical (exhaust, focus_laser, start/monitor_cutting)
        'cooking',          # 14 steps, 4 critical (preheat, saute, cook_thoroughly, turn_off_stove)
        'latte_making'      # 20 steps, 2 critical (brew_coffee, steam_milk)
    ]


def define_cost_regimes() -> Dict[str, Dict[str, Any]]:
    """Define 3 cost regimes with IMPROVED parameters."""
    return {
        'very_high_stakes': {
            'c_fail': 30.0,  # Base cost for failures (× step criticality)
            'f0_base': 0.6,  # 60% baseline failure (INCREASED from 0.3)
            'lambda_forget': 0.05,  # Slow decay (14-tick half-life → 2-3 steps)
            'description': 'Surgery/Critical - failures extremely costly'
        },
        'balanced': {
            'c_fail': 15.0,  # Base cost for failures (× step criticality)
            'f0_base': 0.6,  # 60% baseline failure (INCREASED from 0.3)
            'lambda_forget': 0.05,  # Slow decay (14-tick half-life → 2-3 steps)
            'description': 'Standard assistance - balanced costs'
        },
        'moderate_low': {
            'c_fail': 10.0,  # Base cost for failures (× step criticality)
            'f0_base': 0.6,  # 60% baseline failure (INCREASED from 0.3)
            'lambda_forget': 0.05,  # Slow decay (14-tick half-life → 2-3 steps)
            'description': 'Casual task - low stakes'
        }
    }


def train_single_combination(
    task_name: str,
    regime_name: str,
    regime_config: Dict[str, Any],
    timesteps: int = 200000,
    seed: int = 42
) -> Dict[str, Any]:
    """Train one task-regime combination."""
    start_time = time.time()

    try:
        # Set random seed
        np.random.seed(seed)

        # Get task definition
        task_def = get_task_definition(task_name)

        # Create per-step failure costs (regime base_cost × task step criticality)
        c_fail_per_step = create_per_step_failure_costs(task_def, base_cost=regime_config['c_fail'])

        # Count critical steps (for reporting)
        n_critical = np.sum(c_fail_per_step > 0)

        print(f"\n{'='*80}")
        print(f"TRAINING: {task_name} / {regime_name}")
        print(f"{'='*80}")
        print(f"  Task: {task_name} ({len(task_def.steps)} steps, {n_critical} critical)")
        print(f"  Regime: {regime_name} (c_fail={regime_config['c_fail']}, f0={regime_config['f0_base']})")
        print(f"  Critical steps: {[i for i, c in enumerate(c_fail_per_step) if c > 0]}")
        print(f"  Action space: {2 + n_critical} actions (silent, confirm, + {n_critical} remind_X)")

        # Create SimulationParams with IMPROVED settings
        params = SimulationParams(
            c_fail_per_step=c_fail_per_step,
            c_int=1.0,  # FIXED to 1.0
            c_nar=0.0,   # DISABLED
            c_resp=0.0,  # DISABLED
            f0_base=regime_config['f0_base'],  # 0.6 (60% baseline - INCREASED)
            lambda_forget=regime_config['lambda_forget'],  # 0.05 (14-tick half-life)
            delta_reminder=0.6,  # Strong boost (90-95% immediate prevention)
            k_memory=3.0,  # Steeper curve (emphasizes timing)
            step_mean_duration=8,    # <10 ticks per step (tighter timing)
            step_std_duration=2,     # Lower variance
            obs_noise=0.0,           # DISABLED (no confusion)
            c_off_timing=0.5         # Penalty for reminding wrong step
        )

        # Create save path: models/{regime}/{task}/
        model_dir = PROJECT_ROOT / "models" / regime_name / task_name
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = str(model_dir / "final_model")

        # Train PPO policy
        model = train_ppo_policy(
            params=params,
            task_def=task_def,
            total_timesteps=timesteps,
            save_path=model_path
        )

        elapsed = time.time() - start_time
        print(f"\n✓ SUCCESS: {task_name}/{regime_name} ({elapsed:.1f}s)")

        return {
            'task': task_name,
            'regime': regime_name,
            'status': 'success',
            'elapsed_seconds': elapsed,
            'timesteps': timesteps,
            'n_critical_steps': int(n_critical),
            'model_path': model_path
        }

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ FAILED: {task_name}/{regime_name} - {str(e)}")
        import traceback
        traceback.print_exc()

        return {
            'task': task_name,
            'regime': regime_name,
            'status': 'failed',
            'error': str(e),
            'elapsed_seconds': elapsed
        }


def train_all_combinations(timesteps: int = 200000, seed: int = 42) -> Dict[str, Any]:
    """Train all 21 models sequentially with progress tracking."""
    tasks = get_all_tasks()
    regimes = define_cost_regimes()
    results = []

    total = len(tasks) * len(regimes)
    start_time_all = time.time()

    print(f"\n{'='*80}")
    print(f"CROSS-TASK MULTI-REGIME TRAINING")
    print(f"{'='*80}")
    print(f"  Tasks: {len(tasks)} ({', '.join(tasks)})")
    print(f"  Regimes: {len(regimes)} ({', '.join(regimes.keys())})")
    print(f"  Total models: {total}")
    print(f"  Timesteps per model: {timesteps:,}")
    print(f"  Estimated time: ~{total * 17 / 60:.1f} minutes (~{total * 17 / 3600:.1f} hours)")
    print(f"{'='*80}\n")

    # Train each regime-task combination
    idx = 0
    for regime_name, regime_config in regimes.items():
        print(f"\n{'='*80}")
        print(f"REGIME: {regime_name.upper()}")
        print(f"  {regime_config['description']}")
        print(f"  c_fail={regime_config['c_fail']}, f0_base={regime_config['f0_base']}")
        print(f"{'='*80}\n")

        for task in tasks:
            idx += 1
            print(f"\n[{idx}/{total}] Starting {task} / {regime_name}...")

            result = train_single_combination(
                task_name=task,
                regime_name=regime_name,
                regime_config=regime_config,
                timesteps=timesteps,
                seed=seed
            )
            results.append(result)

            # Save intermediate results every 3 models (for recovery)
            if idx % 3 == 0:
                interim_path = PROJECT_ROOT / "data" / "results" / "cross_task_training_interim.json"
                with open(interim_path, 'w') as f:
                    json.dump({'completed': idx, 'total': total, 'results': results}, f, indent=2)

    # Final statistics
    elapsed_all = time.time() - start_time_all
    n_success = sum(1 for r in results if r['status'] == 'success')
    n_failed = sum(1 for r in results if r['status'] == 'failed')

    summary = {
        'total_models': total,
        'successful': n_success,
        'failed': n_failed,
        'total_time_seconds': elapsed_all,
        'avg_time_per_model_seconds': elapsed_all / total,
        'results': results,
        'config': {
            'timesteps': timesteps,
            'seed': seed,
            'tasks': tasks,
            'regimes': {k: v for k, v in regimes.items()}
        }
    }

    # Save final results
    output_path = PROJECT_ROOT / "data" / "results" / "cross_task_training_summary.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"  Success: {n_success}/{total} models")
    print(f"  Failed: {n_failed}/{total} models")
    print(f"  Total time: {elapsed_all:.1f}s ({elapsed_all/60:.1f} min / {elapsed_all/3600:.1f} hr)")
    print(f"  Avg time per model: {elapsed_all/total:.1f}s")
    print(f"\nResults saved to: {output_path}")

    if n_failed > 0:
        print(f"\n⚠ {n_failed} models failed. Check results above for details.")
        failed_models = [(r['task'], r['regime']) for r in results if r['status'] == 'failed']
        for task, regime in failed_models:
            print(f"  ✗ {task}/{regime}")
    else:
        print(f"\n✓ All models trained successfully!")

    print(f"\nNext steps:")
    print(f"  1. Evaluate: cd ../experiments && python evaluate_cross_task_all_regimes.py")
    print(f"  2. Visualize: cd ../visualization && python plot_reward_curves.py")
    print(f"  3. Report: python generate_cross_task_report.py")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Train RL policies for all tasks across all regimes')
    parser.add_argument('--timesteps', type=int, default=200000,
                       help='Number of timesteps per model (default: 200000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    # Run training
    summary = train_all_combinations(timesteps=args.timesteps, seed=args.seed)

    # Exit with error code if any failed
    if summary['failed'] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
