"""
Cross-Task Multi-Regime Evaluation

Evaluate all 21 trained RL policies against random baseline.
Generates comprehensive comparison metrics.

Usage:
    python evaluate_cross_task_all_regimes.py --n-episodes 100
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training"))

from procedure_assistant_sim import ProcedureAssistantEnv, SimulationParams
from task_definitions import get_task_definition, create_per_step_failure_costs
from train_rl_policy import GymWrapperEnv
from stable_baselines3 import PPO


def get_all_tasks() -> List[str]:
    """Return all 7 task names."""
    return ['make_cereal', 'make_coffee', 'make_tea', 'make_sandwich',
            'make_stencil', 'cooking', 'latte_making']


def define_cost_regimes() -> Dict[str, Dict[str, Any]]:
    """Define 3 cost regimes (same as training)."""
    return {
        'very_high_stakes': {'c_fail': 30.0, 'f0_base': 0.6, 'lambda_forget': 0.05},
        'balanced': {'c_fail': 15.0, 'f0_base': 0.6, 'lambda_forget': 0.05},
        'moderate_low': {'c_fail': 10.0, 'f0_base': 0.6, 'lambda_forget': 0.05}
    }


def evaluate_policy(env: ProcedureAssistantEnv, model: PPO, n_episodes: int = 100) -> Dict[str, Any]:
    """Evaluate a trained PPO model."""
    rewards = []
    interventions = []
    failures = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_interventions = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            if action != 0:  # Not silent
                episode_interventions += 1

        rewards.append(episode_reward)
        interventions.append(episode_interventions)
        failures.append(env.env.pa_state.total_failures)

    return {
        'reward_mean': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards)),
        'interventions_mean': float(np.mean(interventions)),
        'failures_mean': float(np.mean(failures))
    }


def evaluate_random(env: ProcedureAssistantEnv, n_episodes: int = 100) -> Dict[str, Any]:
    """Evaluate random baseline."""
    rewards = []
    interventions = []
    failures = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_interventions = 0
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            if action != 0:
                episode_interventions += 1

        rewards.append(episode_reward)
        interventions.append(episode_interventions)
        failures.append(env.env.pa_state.total_failures)

    return {
        'reward_mean': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards)),
        'interventions_mean': float(np.mean(interventions)),
        'failures_mean': float(np.mean(failures))
    }


def evaluate_single_model(task: str, regime_name: str, regime_config: Dict,
                          n_episodes: int = 100) -> Dict[str, Any]:
    """Evaluate one trained model vs random baseline."""
    print(f"  Evaluating {task} / {regime_name}...")

    # Get task and create params
    task_def = get_task_definition(task)
    c_fail_per_step = create_per_step_failure_costs(task_def, base_cost=regime_config['c_fail'])

    params = SimulationParams(
        c_fail_per_step=c_fail_per_step,
        c_int=1.0,
        c_nar=0.0,
        c_resp=0.0,
        f0_base=regime_config['f0_base'],
        lambda_forget=regime_config['lambda_forget'],
        delta_reminder=0.6,
        k_memory=3.0,
        step_mean_duration=8,
        obs_noise=0.0,
        c_off_timing=0.5
    )

    env = GymWrapperEnv(params, task_def)

    # Load trained model
    model_path = PROJECT_ROOT / "models" / regime_name / task / "final_model" / "final_model.zip"

    if not model_path.exists():
        return {
            'task': task,
            'regime': regime_name,
            'status': 'model_not_found',
            'error': f'Model not found at {model_path}'
        }

    try:
        model = PPO.load(str(model_path))

        # Evaluate RL
        rl_results = evaluate_policy(env, model, n_episodes)

        # Evaluate random
        random_results = evaluate_random(env, n_episodes)

        # Calculate improvement
        if random_results['reward_mean'] != 0:
            improvement = (rl_results['reward_mean'] - random_results['reward_mean']) / abs(random_results['reward_mean']) * 100
        else:
            improvement = 0.0

        return {
            'task': task,
            'regime': regime_name,
            'status': 'success',
            'rl_reward_mean': rl_results['reward_mean'],
            'rl_reward_std': rl_results['reward_std'],
            'rl_interventions_mean': rl_results['interventions_mean'],
            'rl_failures_mean': rl_results['failures_mean'],
            'random_reward_mean': random_results['reward_mean'],
            'random_interventions_mean': random_results['interventions_mean'],
            'random_failures_mean': random_results['failures_mean'],
            'improvement_pct': float(improvement)
        }

    except Exception as e:
        return {
            'task': task,
            'regime': regime_name,
            'status': 'eval_failed',
            'error': str(e)
        }


def evaluate_all_models(n_episodes: int = 100) -> pd.DataFrame:
    """Evaluate all 21 models and create summary table."""
    tasks = get_all_tasks()
    regimes = define_cost_regimes()
    results = []

    print(f"\n{'='*80}")
    print(f"EVALUATING ALL MODELS")
    print(f"{'='*80}")
    print(f"  Total models: {len(tasks) * len(regimes)}")
    print(f"  Episodes per model: {n_episodes}")
    print(f"{'='*80}\n")

    for regime_name, regime_config in regimes.items():
        print(f"\nRegime: {regime_name}")
        for task in tasks:
            result = evaluate_single_model(task, regime_name, regime_config, n_episodes)
            results.append(result)

    df = pd.DataFrame(results)

    # Save to JSON
    output_path = PROJECT_ROOT / "data" / "results" / "cross_task_evaluation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient='records', indent=2)

    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"  Results saved to: {output_path}")
    print(f"{'='*80}\n")

    return df


def main():
    parser = argparse.ArgumentParser(description='Evaluate all trained models')
    parser.add_argument('--n-episodes', type=int, default=100,
                       help='Number of evaluation episodes per model (default: 100)')

    args = parser.parse_args()

    df = evaluate_all_models(n_episodes=args.n_episodes)

    # Print summary stats
    successful = df[df['status'] == 'success']
    if len(successful) > 0:
        print(f"Summary Statistics:")
        print(f"  Mean improvement: {successful['improvement_pct'].mean():.1f}%")
        print(f"  Median improvement: {successful['improvement_pct'].median():.1f}%")
        print(f"  Best: {successful.loc[successful['improvement_pct'].idxmax(), 'task']} / "
              f"{successful.loc[successful['improvement_pct'].idxmax(), 'regime']} "
              f"(+{successful['improvement_pct'].max():.1f}%)")


if __name__ == "__main__":
    main()
