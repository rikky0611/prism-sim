"""
Evaluate Single Task Multi-Regime Models

Evaluate RL policies for one task across 3 cost regimes against baselines.

Usage:
    python evaluate_single_task.py --task latte_making --n-episodes 100
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
from stable_baselines3 import PPO

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training"))

from procedure_assistant_sim import SimulationParams, ProcedureAssistantEnv
from task_definitions import get_task_definition, create_per_step_failure_costs
from train_rl_policy import GymWrapperEnv


def define_cost_regimes():
    """Define 3 cost regimes."""
    return {
        'extremely_high': {
            'c_fail': 50.0,
            'f0_base': 0.6,
            'lambda_forget': 0.03,  # Slower decay - reminders last longer
            'description': 'Ultra-critical scenarios'
        },
        'moderate': {
            'c_fail': 15.0,
            'f0_base': 0.6,
            'lambda_forget': 0.03,  # Slower decay - reminders last longer
            'description': 'Standard'
        },
        'extremely_low': {
            'c_fail': 5.0,
            'f0_base': 0.6,
            'lambda_forget': 0.03,  # Slower decay - reminders last longer
            'description': 'Low-stakes tasks'
        }
    }


def evaluate_policy(env, policy_fn, n_episodes=100):
    """Evaluate a policy over n_episodes."""
    rewards = []
    interruptions = []
    failures = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_interruptions = 0
        episode_failures = 0

        while not done:
            action = policy_fn(obs)
            result = env.step(action)

            # Handle both old (4 values) and new (5 values) Gymnasium API
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result

            episode_reward += reward

            # Count interruptions and failures
            if action != 0:  # Not silent
                episode_interruptions += 1

        # Get final state
        episode_failures = env.env.pa_state.total_failures

        rewards.append(episode_reward)
        interruptions.append(episode_interruptions)
        failures.append(episode_failures)

    return {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'mean_interruptions': float(np.mean(interruptions)),
        'std_interruptions': float(np.std(interruptions)),
        'mean_failures': float(np.mean(failures)),
        'std_failures': float(np.std(failures)),
        'rewards': [float(r) for r in rewards]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='latte_making')
    parser.add_argument('--n-episodes', type=int, default=100)
    args = parser.parse_args()

    print("="*80)
    print(f"EVALUATING: {args.task}")
    print("="*80)

    # Load task
    task_def = get_task_definition(args.task)
    regimes = define_cost_regimes()

    results = {}

    for regime_name, regime_config in regimes.items():
        print(f"\n{regime_name.upper()}:")
        print(f"  c_fail={regime_config['c_fail']}, c_remind=1.0")

        # Create environment
        c_fail_per_step = create_per_step_failure_costs(task_def, base_cost=regime_config['c_fail'])
        params = SimulationParams(
            c_fail_per_step=c_fail_per_step,
            c_remind=1.0,
            f0_base=regime_config['f0_base'],
            lambda_forget=regime_config['lambda_forget']
        )
        gym_env = GymWrapperEnv(params, task_def)

        # Evaluate Random baseline
        print("  Evaluating Random...", end='', flush=True)
        random_results = evaluate_policy(
            gym_env,
            lambda obs: np.random.randint(gym_env.action_space.n),
            n_episodes=args.n_episodes
        )
        print(f" Reward: {random_results['mean_reward']:.1f}")

        # Evaluate RL model
        model_path = PROJECT_ROOT / "models" / regime_name / args.task / "final_model" / "final_model.zip"
        if model_path.exists():
            print(f"  Evaluating RL PPO...", end='', flush=True)
            model = PPO.load(str(model_path))
            rl_results = evaluate_policy(
                gym_env,
                lambda obs: model.predict(obs, deterministic=True)[0],
                n_episodes=args.n_episodes
            )
            print(f" Reward: {rl_results['mean_reward']:.1f}")

            # Calculate improvement
            improvement_pct = ((rl_results['mean_reward'] - random_results['mean_reward']) /
                             abs(random_results['mean_reward']) * 100)

            results[regime_name] = {
                'Random': random_results,
                'RL_PPO': rl_results,
                'improvement_pct': improvement_pct
            }
        else:
            print(f"  ✗ Model not found: {model_path}")
            results[regime_name] = {
                'Random': random_results,
                'error': 'Model not found'
            }

    # Save results
    output_path = PROJECT_ROOT / "data" / "results" / f"evaluation_{args.task}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n{'Regime':<20} {'Random':<15} {'RL PPO':<15} {'Improvement':<15}")
    print("-"*65)

    for regime_name in ['very_high_stakes', 'balanced', 'moderate_low']:
        if regime_name in results and 'RL_PPO' in results[regime_name]:
            random_reward = results[regime_name]['Random']['mean_reward']
            rl_reward = results[regime_name]['RL_PPO']['mean_reward']
            improvement = results[regime_name]['improvement_pct']

            print(f"{regime_name:<20} {random_reward:<15.1f} {rl_reward:<15.1f} {improvement:>+6.1f}%")

            # Show interventions
            random_ints = results[regime_name]['Random']['mean_interruptions']
            rl_ints = results[regime_name]['RL_PPO']['mean_interruptions']
            print(f"  {'Interruptions:':<20} {random_ints:<15.2f} {rl_ints:<15.2f}")

            # Show failures
            random_fails = results[regime_name]['Random']['mean_failures']
            rl_fails = results[regime_name]['RL_PPO']['mean_failures']
            print(f"  {'Failures:':<20} {random_fails:<15.2f} {rl_fails:<15.2f}")
            print()

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
