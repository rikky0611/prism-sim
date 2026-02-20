"""
Cross-Task Multi-Regime RL Training

Train RL policies across 7 procedural tasks with 3 different cost regimes.
Total: 21 models (7 tasks × 3 regimes)

Usage:
    python train_cross_task_multi_regime.py --timesteps 50000
    python train_cross_task_multi_regime.py --timesteps 50000 --resume
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import sys
import numpy as np
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training"))

from procedure_assistant_sim import SimulationParams
from task_definitions import TaskDefinition, load_task_definitions, create_per_step_failure_costs
from train_rl_policy import train_ppo_policy


def define_cost_regimes() -> Dict[str, Dict[str, Any]]:
    """Define 3 cost regimes with clear naming.

    NOTE: c_int is now FIXED to 1.0 (in SimulationParams).
    c_fail is used as base_cost for per-step failure costs (× step criticality).

    Returns:
        Dictionary mapping regime names to configuration dicts.
    """
    return {
        'extremely_high': {
            'c_fail': 50.0,  # Base cost for failures (× step criticality)
            'f0_base': 0.6,
            'lambda_forget': 0.03,  # Slower decay - reminders last longer
            'description': 'Ultra-critical scenarios - failures catastrophic (c_int=1 fixed)'
        },
        'moderate': {
            'c_fail': 15.0,  # Base cost for failures (× step criticality)
            'f0_base': 0.6,
            'lambda_forget': 0.03,  # Slower decay - reminders last longer
            'description': 'Standard assistance - balanced costs (c_int=1 fixed)'
        },
        'extremely_low': {
            'c_fail': 5.0,  # Base cost for failures (× step criticality)
            'f0_base': 0.6,
            'lambda_forget': 0.03,  # Slower decay - reminders last longer
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
    """Train a single PPO model for one task-regime combination.

    Args:
        task_name: Name of the task
        task_def: TaskDefinition object
        regime_name: Name of the cost regime
        regime_config: Cost regime configuration dict
        timesteps: Number of training timesteps
        seed: Random seed for reproducibility

    Returns:
        Training metadata dict (duration, model_path, params, success)
    """
    start_time = time.time()

    try:
        # Set random seed for reproducibility
        np.random.seed(seed)

        # Create per-step failure costs (regime base_cost × task step criticality)
        c_fail_per_step = create_per_step_failure_costs(task_def, base_cost=regime_config['c_fail'])

        # Create SimulationParams with NEW cost structure
        params = SimulationParams(
            c_fail_per_step=c_fail_per_step,
            c_int=1.0,  # FIXED to 1.0
            f0_base=regime_config['f0_base'],
            lambda_forget=regime_config['lambda_forget'],
            delta_reminder=0.8  # Stronger reminder boost (was 0.6)
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

        duration = time.time() - start_time

        return {
            'success': True,
            'task_name': task_name,
            'regime_name': regime_name,
            'model_path': model_path,
            'timesteps': timesteps,
            'duration_seconds': duration,
            'regime_config': regime_config,
            'task_config': {
                'n_steps': task_def.n_steps,
                'domain': task_def.domain
            }
        }

    except Exception as e:
        duration = time.time() - start_time
        return {
            'success': False,
            'task_name': task_name,
            'regime_name': regime_name,
            'error': str(e),
            'duration_seconds': duration
        }


def train_all_combinations(
    timesteps_per_model: int = 50000,
    resume: bool = False,
    checkpoint_frequency: int = 3
) -> Dict[str, Any]:
    """Train all 21 models (7 tasks × 3 regimes).

    Args:
        timesteps_per_model: Training timesteps for each model
        resume: If True, skip already trained models
        checkpoint_frequency: Save checkpoint every N models

    Returns:
        Complete training summary dict
    """
    # Load all task definitions
    tasks = load_task_definitions()
    task_order = [
        'make_cereal', 'make_coffee', 'make_tea', 'make_sandwich',
        'cooking', 'make_stencil', 'latte_making'
    ]

    # Define cost regimes
    regimes = define_cost_regimes()
    regime_order = ['extremely_high', 'moderate', 'extremely_low']

    # Initialize results storage
    training_results = {
        'metadata': {
            'experiment_type': 'cross_task_multi_regime',
            'n_tasks': len(task_order),
            'n_regimes': len(regime_order),
            'n_models': len(task_order) * len(regime_order),
            'timesteps_per_model': timesteps_per_model,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'resume_mode': resume
        },
        'regimes': regimes,
        'tasks': {
            name: {
                'n_steps': tasks[name].n_steps,
                'domain': tasks[name].domain
            } for name in task_order
        },
        'training_log': []
    }

    # Setup progress tracking
    total_models = len(task_order) * len(regime_order)
    pbar = tqdm(total=total_models, desc="Training Progress")
    start_time = time.time()
    completed_count = 0

    # Train all combinations
    for regime_name in regime_order:
        regime_config = regimes[regime_name]

        for task_name in task_order:
            task_def = tasks[task_name]

            # Check if model already exists (resume mode)
            model_path = PROJECT_ROOT / "models" / regime_name / task_name / "final_model.zip"
            if resume and model_path.exists():
                pbar.update(1)
                pbar.set_postfix({
                    'regime': regime_name,
                    'task': task_name,
                    'status': 'SKIPPED'
                })
                training_results['training_log'].append({
                    'task_name': task_name,
                    'regime_name': regime_name,
                    'skipped': True
                })
                completed_count += 1
                continue

            # Train model
            pbar.set_postfix({
                'regime': regime_name,
                'task': task_name,
                'status': 'TRAINING'
            })

            result = train_single_model(
                task_name=task_name,
                task_def=task_def,
                regime_name=regime_name,
                regime_config=regime_config,
                timesteps=timesteps_per_model
            )

            training_results['training_log'].append(result)
            completed_count += 1

            # Update progress bar
            pbar.update(1)
            elapsed = time.time() - start_time
            avg_time_per_model = elapsed / completed_count
            remaining_models = total_models - completed_count
            eta_seconds = avg_time_per_model * remaining_models

            status = '✓ SUCCESS' if result['success'] else '✗ FAILED'
            pbar.set_postfix({
                'regime': regime_name,
                'task': task_name,
                'status': status,
                'ETA': f"{eta_seconds/60:.1f}m"
            })

            # Save checkpoint
            if completed_count % checkpoint_frequency == 0:
                checkpoint_path = PROJECT_ROOT / "data" / "results" / "cross_task_training_checkpoint.json"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_path, 'w') as f:
                    json.dump(training_results, f, indent=2)

    pbar.close()

    # Add final metadata
    total_duration = time.time() - start_time
    training_results['metadata']['total_training_duration_seconds'] = total_duration
    training_results['metadata']['total_training_duration_minutes'] = total_duration / 60

    # Count successes and failures
    successes = sum(1 for r in training_results['training_log'] if r.get('success', False))
    failures = sum(1 for r in training_results['training_log'] if not r.get('success', True) and not r.get('skipped', False))
    skipped = sum(1 for r in training_results['training_log'] if r.get('skipped', False))

    training_results['metadata']['successful_models'] = successes
    training_results['metadata']['failed_models'] = failures
    training_results['metadata']['skipped_models'] = skipped

    return training_results


def main():
    """Main entry point for cross-task multi-regime training."""
    parser = argparse.ArgumentParser(
        description='Train RL policies across 7 tasks and 3 cost regimes'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=50000,
        help='Training timesteps per model (default: 50000)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training by skipping existing models'
    )
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=3,
        help='Save checkpoint every N models (default: 3)'
    )

    args = parser.parse_args()

    print("="*70)
    print("CROSS-TASK MULTI-REGIME RL TRAINING")
    print("="*70)
    print(f"Tasks: 7 (make_cereal, make_coffee, make_tea, make_sandwich,")
    print(f"        cooking, make_stencil, latte_making)")
    print(f"Regimes: 3 (very_high_stakes, balanced, moderate_low)")
    print(f"Total models: 21")
    print(f"Timesteps per model: {args.timesteps:,}")
    print(f"Resume mode: {args.resume}")
    print(f"Checkpoint frequency: every {args.checkpoint_freq} models")
    print(f"Estimated duration: ~{21 * args.timesteps / 50000 * 3:.0f}-{21 * args.timesteps / 50000 * 4:.0f} minutes")
    print("="*70)
    print()

    # Run training
    results = train_all_combinations(
        timesteps_per_model=args.timesteps,
        resume=args.resume,
        checkpoint_frequency=args.checkpoint_freq
    )

    # Save final results
    output_path = PROJECT_ROOT / "data" / "results" / "cross_task_multi_regime_training.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print("="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Total duration: {results['metadata']['total_training_duration_minutes']:.1f} minutes")
    print(f"Successful models: {results['metadata']['successful_models']}")
    print(f"Failed models: {results['metadata']['failed_models']}")
    print(f"Skipped models: {results['metadata']['skipped_models']}")
    print(f"\nResults saved to: {output_path}")
    print(f"Models saved to: {PROJECT_ROOT}/models/")
    print("="*70)


if __name__ == '__main__':
    main()
