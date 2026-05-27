"""
Single Task Multi-Regime Training

Train RL policies for ONE task across 3 cost regimes (for testing).
Total: 3 models

Usage:
    python train_single_task_multi_regime.py --task latte_making --timesteps 50000
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any
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


def define_cost_regimes() -> Dict[str, Dict[str, Any]]:
    """Define 3 cost regimes with SIMPLIFIED memory model (single component)."""
    return {
        'extremely_high': {
            'c_fail': 50.0,  # Base cost for failures (× step criticality)
            'f0_base': 0.3,  # 30% baseline failure probability
            'lambda_forget': 0.03,  # Slower decay (23-tick half-life) - reminders last longer
            'description': 'Ultra-critical scenarios - failures catastrophic (c_int=1 fixed)'
        },
        'moderate': {
            'c_fail': 15.0,  # Base cost for failures (× step criticality)
            'f0_base': 0.3,  # 30% baseline failure probability
            'lambda_forget': 0.03,  # Slower decay (23-tick half-life) - reminders last longer
            'description': 'Standard assistance - balanced costs (c_int=1 fixed)'
        },
        'extremely_low': {
            'c_fail': 5.0,  # Base cost for failures (× step criticality)
            'f0_base': 0.3,  # 30% baseline failure probability
            'lambda_forget': 0.03,  # Slower decay (23-tick half-life) - reminders last longer
            'description': 'Low-stakes tasks - failures minor (c_int=1 fixed)'
        }
    }


def train_single_model(
    task_name: str,
    task_def: TaskDefinition,
    regime_name: str,
    regime_config: Dict[str, Any],
    timesteps: int = 50000,
    seed: int = 42
) -> Dict[str, Any]:
    """Train a single PPO model for one task-regime combination."""
    start_time = time.time()

    try:
        # Set random seed
        np.random.seed(seed)

        # Create per-step failure costs (regime base_cost × task step criticality)
        c_fail_per_step = create_per_step_failure_costs(task_def, base_cost=regime_config['c_fail'])

        print(f"\nPer-step costs for {task_name} ({regime_name}):")
        print(f"  Base cost: {regime_config['c_fail']}")
        print(f"  Per-step costs: {c_fail_per_step[:5]}... (first 5 steps)")

        # Create SimulationParams with SIMPLIFIED memory model (single component)
        params = SimulationParams(
            c_fail_per_step=c_fail_per_step,
            c_int=1.0,  # FIXED to 1.0
            c_nar=0.0,   # DISABLED: No cost for human narrations (simplicity)
            c_resp=0.0,  # DISABLED: No cost for human responses (simplicity)
            f0_base=regime_config['f0_base'],  # 0.3 (30% baseline)
            lambda_forget=regime_config['lambda_forget'],  # 0.03 (23-tick half-life)
            delta_reminder=0.8,  # Stronger reminder boost (was 0.6)
            k_memory=3.0,  # Steeper curve (emphasizes timing)
            step_mean_duration=8,    # <10 ticks per step (was 30) - tighter timing
            step_std_duration=2,     # Lower variance (was 10)
            obs_noise=0.0,           # DISABLED: No observation noise (eliminate confusion)
            c_off_timing=0.5         # Penalty for reminding wrong step (50% of base cost)
        )

        # Create save path: models/{regime}/{task}/
        model_dir = PROJECT_ROOT / "models" / regime_name / task_name
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = str(model_dir / "final_model")

        # Train PPO policy
        print(f"\nTraining {task_name} / {regime_name}...")
        model = train_ppo_policy(
            params=params,
            task_def=task_def,
            total_timesteps=timesteps,
            save_path=model_path
        )

        duration = time.time() - start_time

        return {
            'success': True,
            'task_name': task_name,
            'regime_name': regime_name,
            'model_path': model_path,
            'duration_sec': duration,
            'timesteps': timesteps,
            'regime_config': regime_config
        }

    except Exception as e:
        duration = time.time() - start_time
        print(f"\n❌ ERROR training {task_name}/{regime_name}: {e}")
        return {
            'success': False,
            'task_name': task_name,
            'regime_name': regime_name,
            'error': str(e),
            'duration_sec': duration
        }


def main():
    parser = argparse.ArgumentParser(description='Train single task across 3 regimes')
    parser.add_argument('--task', type=str, default='latte_making',
                       help='Task to train (default: latte_making)')
    parser.add_argument('--timesteps', type=int, default=50000,
                       help='Training timesteps per model (default: 50000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    args = parser.parse_args()

    print("="*80)
    print("SINGLE TASK MULTI-REGIME TRAINING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Task: {args.task}")
    print(f"  Timesteps per model: {args.timesteps:,}")
    print(f"  Total models: 3 (3 regimes)")
    print(f"  Estimated time: ~{3 * args.timesteps / 50000 * 60:.0f} seconds")

    # Load task definition
    try:
        task_def = get_task_definition(args.task)
        print(f"\n✓ Loaded task: {args.task} ({task_def.n_steps} steps)")
    except KeyError as e:
        print(f"\n❌ ERROR: Unknown task '{args.task}'")
        print(f"Available tasks: make_cereal, make_coffee, make_tea, make_sandwich, cooking, make_stencil, latte_making")
        return

    # Define cost regimes
    regimes = define_cost_regimes()
    print(f"✓ Defined {len(regimes)} cost regimes")

    # Train all 3 models
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)

    results = []
    start_time = time.time()

    for regime_name, regime_config in tqdm(regimes.items(), desc="Regimes"):
        result = train_single_model(
            task_name=args.task,
            task_def=task_def,
            regime_name=regime_name,
            regime_config=regime_config,
            timesteps=args.timesteps,
            seed=args.seed
        )
        results.append(result)

        if result['success']:
            print(f"  ✓ {regime_name}: {result['duration_sec']:.1f}s")
        else:
            print(f"  ✗ {regime_name}: FAILED")

    total_duration = time.time() - start_time

    # Save results
    output_path = PROJECT_ROOT / "data" / "results" / f"single_task_training_{args.task}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'task': args.task,
        'n_steps': task_def.n_steps,
        'timesteps_per_model': args.timesteps,
        'total_models': len(regimes),
        'results': results,
        'total_duration_sec': total_duration
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    success_count = sum(1 for r in results if r['success'])
    print(f"\nResults:")
    print(f"  Success: {success_count}/{len(results)} models")
    print(f"  Total time: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"  Avg time per model: {total_duration/len(results):.1f}s")
    print(f"\nResults saved to: {output_path}")

    if success_count == len(results):
        print(f"\n✓ All models trained successfully!")
        print(f"\nNext steps:")
        print(f"  1. Evaluate: cd ../experiments && python evaluate_single_task.py --task {args.task}")
        print(f"  2. Visualize results")
    else:
        print(f"\n⚠ Some models failed. Check results above.")


if __name__ == "__main__":
    main()
