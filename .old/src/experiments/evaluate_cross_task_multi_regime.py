"""
Cross-Task Multi-Regime RL Evaluation

Evaluate all 21 trained RL models against baseline policies.
Compare performance across tasks and cost regimes.

Usage:
    python evaluate_cross_task_multi_regime.py --n-episodes 100
"""

import argparse
import json
import gc
import time
from pathlib import Path
from typing import Dict, Any, Optional
import sys
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training"))

from procedure_assistant_sim import (
    ProcedureAssistantEnv,
    SimulationParams,
    RandomAssistantPolicy,
    ProactiveReminderPolicy,
    ReactivePolicyHighCost
)
from task_definitions import TaskDefinition, load_task_definitions, create_per_step_failure_costs
from train_rl_policy import evaluate_policy


def load_trained_model(
    task_name: str,
    regime_name: str,
    model_dir: Path
) -> Optional[PPO]:
    """Load a trained PPO model.

    Args:
        task_name: Name of the task
        regime_name: Name of the cost regime
        model_dir: Root models directory

    Returns:
        Loaded PPO model, or None if model doesn't exist
    """
    # Check both possible paths (with and without subdirectory)
    model_path1 = model_dir / regime_name / task_name / "final_model.zip"
    model_path2 = model_dir / regime_name / task_name / "final_model" / "final_model.zip"

    model_path = model_path2 if model_path2.exists() else model_path1

    if not model_path.exists():
        print(f"Warning: Model not found at {model_path}")
        return None

    try:
        model = PPO.load(str(model_path))
        return model
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None


def evaluate_single_model(
    task_name: str,
    task_def: TaskDefinition,
    regime_name: str,
    regime_config: Dict[str, Any],
    model: Optional[PPO],
    n_episodes: int = 100,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """Evaluate one model against baselines.

    Args:
        task_name: Name of the task
        task_def: TaskDefinition object
        regime_name: Name of the cost regime
        regime_config: Cost regime configuration dict
        model: Trained PPO model (or None if not available)
        n_episodes: Number of evaluation episodes
        seed: Random seed

    Returns:
        Dict mapping policy names to metrics dicts
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Create per-step failure costs (regime base_cost × task step criticality)
    c_fail_per_step = create_per_step_failure_costs(task_def, base_cost=regime_config['c_fail'])

    # Create environment with regime params
    params = SimulationParams(
        c_fail_per_step=c_fail_per_step,
        c_int=regime_config.get('c_int', 1.0),
        f0_base=regime_config['f0_base'],
        lambda_forget=regime_config['lambda_forget']
    )

    env = ProcedureAssistantEnv(params, task_def)

    # Evaluate baseline policies
    results = {}

    # 1. Random policy
    random_policy = RandomAssistantPolicy(task_def.n_steps)
    results['Random'] = evaluate_policy(
        env=env,
        policy=random_policy,
        task_def=task_def,
        n_episodes=n_episodes,
        seed=seed
    )

    # 2. Proactive policy
    proactive_policy = ProactiveReminderPolicy(
        task_def.n_steps,
        memory_threshold=0.3,
        lookahead=1
    )
    results['Proactive'] = evaluate_policy(
        env=env,
        policy=proactive_policy,
        task_def=task_def,
        n_episodes=n_episodes,
        seed=seed + 1
    )

    # 3. Reactive policy
    reactive_policy = ReactivePolicyHighCost(
        task_def.n_steps,
        risk_threshold=0.30,
        params=params
    )
    results['Reactive'] = evaluate_policy(
        env=env,
        policy=reactive_policy,
        task_def=task_def,
        n_episodes=n_episodes,
        seed=seed + 2
    )

    # 4. RL policy (if available)
    if model is not None:
        results['RL_PPO'] = evaluate_policy(
            env=env,
            policy=model,
            task_def=task_def,
            n_episodes=n_episodes,
            seed=seed + 3
        )
    else:
        results['RL_PPO'] = {
            'mean_reward': None,
            'std_reward': None,
            'mean_interruptions': None,
            'std_interruptions': None,
            'mean_failures': None,
            'std_failures': None,
            'error': 'Model not found'
        }

    return results


def compute_improvement_metrics(policy_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Compute improvement metrics for RL policy vs baselines.

    Args:
        policy_results: Dict mapping policy names to metrics

    Returns:
        Dict with improvement statistics
    """
    if policy_results['RL_PPO']['mean_reward'] is None:
        return {
            'improvement_pct': None,
            'improvement_absolute': None,
            'best_baseline': None,
            'best_baseline_reward': None
        }

    # Find best baseline
    baseline_names = ['Random', 'Proactive', 'Reactive']
    baseline_rewards = {
        name: policy_results[name]['mean_reward']
        for name in baseline_names
    }
    best_baseline = max(baseline_rewards, key=baseline_rewards.get)
    best_baseline_reward = baseline_rewards[best_baseline]

    # Compute improvement (remember: rewards are negative costs)
    rl_reward = policy_results['RL_PPO']['mean_reward']
    improvement_absolute = rl_reward - best_baseline_reward
    improvement_pct = (improvement_absolute / abs(best_baseline_reward)) * 100

    return {
        'improvement_pct': improvement_pct,
        'improvement_absolute': improvement_absolute,
        'best_baseline': best_baseline,
        'best_baseline_reward': best_baseline_reward,
        'rl_reward': rl_reward
    }


def evaluate_all_models(
    n_episodes: int = 100,
    seed: int = 42
) -> Dict[str, Any]:
    """Evaluate all 21 trained models.

    Args:
        n_episodes: Number of evaluation episodes per model
        seed: Random seed

    Returns:
        Complete evaluation results dict
    """
    # Load training summary to get regimes and tasks
    training_path = PROJECT_ROOT / "data" / "results" / "cross_task_multi_regime_training.json"
    if not training_path.exists():
        raise FileNotFoundError(
            f"Training summary not found at {training_path}. "
            "Run train_cross_task_multi_regime.py first."
        )

    with open(training_path, 'r') as f:
        training_summary = json.load(f)

    regimes = training_summary['regimes']
    tasks_info = training_summary['tasks']
    task_names = list(tasks_info.keys())

    # Load task definitions
    all_tasks = load_task_definitions()

    # Initialize results structure
    evaluation_results = {
        'metadata': {
            'experiment_type': 'cross_task_multi_regime_evaluation',
            'n_tasks': len(task_names),
            'n_regimes': len(regimes),
            'n_models': len(task_names) * len(regimes),
            'n_eval_episodes': n_episodes,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'regimes': regimes,
        'tasks': tasks_info,
        'results': {}
    }

    # Setup progress tracking
    total_evaluations = len(regimes) * len(task_names)
    pbar = tqdm(total=total_evaluations, desc="Evaluation Progress")

    models_dir = PROJECT_ROOT / "models"

    # Evaluate all combinations
    for regime_name, regime_config in regimes.items():
        evaluation_results['results'][regime_name] = {}

        for task_name in task_names:
            task_def = all_tasks[task_name]

            # Load trained model
            model = load_trained_model(task_name, regime_name, models_dir)

            # Evaluate all policies
            policy_results = evaluate_single_model(
                task_name=task_name,
                task_def=task_def,
                regime_name=regime_name,
                regime_config=regime_config,
                model=model,
                n_episodes=n_episodes,
                seed=seed
            )

            # Compute improvement metrics
            improvement_metrics = compute_improvement_metrics(policy_results)

            # Store results
            evaluation_results['results'][regime_name][task_name] = {
                'policies': policy_results,
                **improvement_metrics
            }

            # Update progress
            pbar.update(1)
            pbar.set_postfix({
                'regime': regime_name,
                'task': task_name,
                'improvement': f"{improvement_metrics.get('improvement_pct', 0):.1f}%"
            })

            # Free memory
            if model is not None:
                del model
            gc.collect()

    pbar.close()

    return evaluation_results


def compute_aggregate_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute aggregate statistics across tasks and regimes.

    Args:
        results: Evaluation results dict

    Returns:
        Dict with aggregate statistics
    """
    import time

    evaluation_data = results['results']
    regimes = list(evaluation_data.keys())
    tasks = list(evaluation_data[regimes[0]].keys())

    # By regime aggregates
    by_regime = {}
    for regime in regimes:
        improvements = []
        rl_rewards = []
        rl_interruptions = []
        rl_failures = []
        n_improved = 0

        for task in tasks:
            task_data = evaluation_data[regime][task]
            if task_data.get('improvement_pct') is not None:
                improvements.append(task_data['improvement_pct'])
                if task_data['improvement_pct'] > 0:
                    n_improved += 1

                rl_policy_data = task_data['policies']['RL_PPO']
                rl_rewards.append(rl_policy_data['mean_reward'])
                rl_interruptions.append(rl_policy_data['mean_interruptions'])
                rl_failures.append(rl_policy_data['mean_failures'])

        by_regime[regime] = {
            'mean_improvement_pct': np.mean(improvements) if improvements else None,
            'median_improvement_pct': np.median(improvements) if improvements else None,
            'std_improvement_pct': np.std(improvements) if improvements else None,
            'mean_rl_reward': np.mean(rl_rewards) if rl_rewards else None,
            'mean_rl_interruptions': np.mean(rl_interruptions) if rl_interruptions else None,
            'mean_rl_failures': np.mean(rl_failures) if rl_failures else None,
            'n_tasks_improved': n_improved,
            'success_rate': n_improved / len(tasks) if tasks else 0
        }

    # By task aggregates
    by_task = {}
    for task in tasks:
        improvements = []
        rl_rewards = []
        regime_performance = {}

        for regime in regimes:
            task_data = evaluation_data[regime][task]
            if task_data.get('improvement_pct') is not None:
                improvements.append(task_data['improvement_pct'])
                rl_reward = task_data['policies']['RL_PPO']['mean_reward']
                rl_rewards.append(rl_reward)
                regime_performance[regime] = rl_reward

        best_regime = max(regime_performance, key=regime_performance.get) if regime_performance else None
        worst_regime = min(regime_performance, key=regime_performance.get) if regime_performance else None

        by_task[task] = {
            'mean_improvement_pct': np.mean(improvements) if improvements else None,
            'std_improvement_pct': np.std(improvements) if improvements else None,
            'mean_rl_reward': np.mean(rl_rewards) if rl_rewards else None,
            'best_regime': best_regime,
            'worst_regime': worst_regime,
            'regime_performance': regime_performance
        }

    # Overall aggregates
    all_improvements = []
    all_rl_rewards = []
    best_case = None
    worst_case = None
    best_improvement = float('-inf')
    worst_improvement = float('inf')

    for regime in regimes:
        for task in tasks:
            task_data = evaluation_data[regime][task]
            if task_data.get('improvement_pct') is not None:
                improvement = task_data['improvement_pct']
                all_improvements.append(improvement)

                rl_reward = task_data['policies']['RL_PPO']['mean_reward']
                all_rl_rewards.append(rl_reward)

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_case = {
                        'regime': regime,
                        'task': task,
                        'improvement_pct': improvement,
                        'rl_reward': rl_reward
                    }

                if improvement < worst_improvement:
                    worst_improvement = improvement
                    worst_case = {
                        'regime': regime,
                        'task': task,
                        'improvement_pct': improvement,
                        'rl_reward': rl_reward
                    }

    overall = {
        'mean_improvement_pct': np.mean(all_improvements) if all_improvements else None,
        'median_improvement_pct': np.median(all_improvements) if all_improvements else None,
        'std_improvement_pct': np.std(all_improvements) if all_improvements else None,
        'mean_rl_reward': np.mean(all_rl_rewards) if all_rl_rewards else None,
        'n_models_evaluated': len(all_improvements),
        'n_models_improved': sum(1 for imp in all_improvements if imp > 0),
        'success_rate': sum(1 for imp in all_improvements if imp > 0) / len(all_improvements) if all_improvements else 0,
        'best_case': best_case,
        'worst_case': worst_case
    }

    return {
        'by_regime': by_regime,
        'by_task': by_task,
        'overall': overall
    }


def main():
    """Main entry point for cross-task multi-regime evaluation."""
    import time

    parser = argparse.ArgumentParser(
        description='Evaluate RL policies across 7 tasks and 3 cost regimes'
    )
    parser.add_argument(
        '--n-episodes',
        type=int,
        default=100,
        help='Number of evaluation episodes per model (default: 100)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    print("="*70)
    print("CROSS-TASK MULTI-REGIME RL EVALUATION")
    print("="*70)
    print(f"Models to evaluate: 21 (7 tasks × 3 regimes)")
    print(f"Evaluation episodes per model: {args.n_episodes}")
    print(f"Baseline policies: Random, Proactive, Reactive")
    print(f"Random seed: {args.seed}")
    print("="*70)
    print()

    # Run evaluation
    results = evaluate_all_models(n_episodes=args.n_episodes, seed=args.seed)

    # Compute aggregate statistics
    print("\nComputing aggregate statistics...")
    aggregate_stats = compute_aggregate_statistics(results)
    results['aggregate_stats'] = aggregate_stats

    # Save results
    output_path = PROJECT_ROOT / "data" / "results" / "cross_task_multi_regime_evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print("="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Overall mean improvement: {aggregate_stats['overall']['mean_improvement_pct']:.2f}%")
    print(f"Overall median improvement: {aggregate_stats['overall']['median_improvement_pct']:.2f}%")
    print(f"Success rate: {aggregate_stats['overall']['success_rate']*100:.1f}%")
    print(f"  ({aggregate_stats['overall']['n_models_improved']}/{aggregate_stats['overall']['n_models_evaluated']} models)")
    print()
    print("Best case:")
    if aggregate_stats['overall']['best_case']:
        best = aggregate_stats['overall']['best_case']
        print(f"  {best['regime']} / {best['task']}: {best['improvement_pct']:.2f}% improvement")
    print()
    print("Worst case:")
    if aggregate_stats['overall']['worst_case']:
        worst = aggregate_stats['overall']['worst_case']
        print(f"  {worst['regime']} / {worst['task']}: {worst['improvement_pct']:.2f}% improvement")
    print()
    print(f"Results saved to: {output_path}")
    print("="*70)


if __name__ == '__main__':
    main()
