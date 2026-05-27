"""
Train RL policy for procedure assistant using PPO
Compare with baseline heuristics (Random, Proactive, Reactive)
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import json
from typing import Dict, Any, Tuple
import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))

from procedure_assistant_sim import (
    ProcedureAssistantEnv,
    SimulationParams,
    RandomAssistantPolicy,
    ProactiveReminderPolicy,
    ReactivePolicyHighCost,
)

from task_definitions import TaskDefinition, load_task_definitions


class GymWrapperEnv(gym.Env):
    """Gymnasium wrapper for ProcedureAssistantEnv.

    Supports dynamic observation and action spaces based on task size.

    Args:
        params: SimulationParams with memory/failure/cost parameters
        task_def: TaskDefinition specifying the procedural task
    """

    def __init__(self, params: SimulationParams, task_def: TaskDefinition):
        super().__init__()
        self.task_def = task_def
        self.n_steps = task_def.n_steps

        # NOTE: No longer calling apply_task_defaults() - c_fail_per_step set directly
        # Ensure c_fail_per_step matches task n_steps
        if len(params.c_fail_per_step) != self.n_steps:
            raise ValueError(f"c_fail_per_step length ({len(params.c_fail_per_step)}) "
                           f"must match task n_steps ({self.n_steps})")

        self.env = ProcedureAssistantEnv(params, task_def)
        self.params = params

        # Define observation space (dynamic size based on task)
        # Observation: (step_estimate, elapsed_time, memory[0], ..., memory[N-1])
        obs_low = np.array([0, 0] + [0.0] * self.n_steps)
        obs_high = np.array([self.n_steps, 200] + [2.0] * self.n_steps)
        self.observation_space = gym.spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )

        # Define action space (dynamic size based on task)
        # Actions: SILENT (0), CONFIRM (1), REMIND_0 (2), ..., REMIND_N-1
        self.action_space = gym.spaces.Discrete(2 + self.n_steps)

        self._episode_reward = 0
        self._episode_interruptions = 0
        self._episode_failures = 0
        self._episode_history = {
            'rewards': [],
            'interruptions': [],
            'failures': []
        }

    def _convert_observation(self, obs: Dict[str, Any]) -> np.ndarray:
        """Convert environment observation to numpy array"""
        step_est = obs['step_estimate']
        elapsed = obs['elapsed_time']
        memory = obs['memory']
        return np.array([step_est, elapsed] + list(memory), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        obs = self.env.reset()
        self._episode_reward = 0
        self._episode_interruptions = 0
        self._episode_failures = 0
        return self._convert_observation(obs), {}

    def step(self, action: int):
        # Action is already an integer index into assistant actions
        obs, reward, done, info = self.env.step(action)

        # Track episode statistics
        self._episode_reward += reward
        if action != 0:  # 0 is always 'silent' action
            self._episode_interruptions += 1
        if info.get('failure', False):
            self._episode_failures += 1

        if done:
            # Save episode stats
            self._episode_history['rewards'].append(self._episode_reward)
            self._episode_history['interruptions'].append(self._episode_interruptions)
            self._episode_history['failures'].append(self._episode_failures)

        return self._convert_observation(obs), reward, done, False, info

    def get_episode_stats(self) -> Dict[str, Any]:
        """Get statistics from completed episodes"""
        if not self._episode_history['rewards']:
            return {}

        return {
            'mean_reward': np.mean(self._episode_history['rewards']),
            'std_reward': np.std(self._episode_history['rewards']),
            'mean_interruptions': np.mean(self._episode_history['interruptions']),
            'mean_failures': np.mean(self._episode_history['failures']),
            'n_episodes': len(self._episode_history['rewards'])
        }


def evaluate_policy(
    env: ProcedureAssistantEnv,
    policy,
    task_def: TaskDefinition,
    n_episodes: int = 100,
    seed: int = 42
) -> Dict[str, Any]:
    """Evaluate a policy on the environment.

    Args:
        env: ProcedureAssistantEnv to evaluate on
        policy: Policy to evaluate (RL or heuristic)
        task_def: TaskDefinition for the task
        n_episodes: Number of evaluation episodes
        seed: Random seed

    Returns:
        Dictionary with evaluation metrics
    """
    np.random.seed(seed)

    rewards = []
    interruptions = []
    failures = []

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0

        while not done:
            if hasattr(policy, 'predict'):
                # RL policy (stable-baselines3)
                wrapper = GymWrapperEnv(env.params, task_def)
                wrapper.env = env
                obs_array = wrapper._convert_observation(obs)
                action, _ = policy.predict(obs_array, deterministic=True)
            else:
                # Heuristic policy
                action = policy.get_action(obs)

            obs, reward, done, info = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)
        interruptions.append(env.pa_state.total_interactions)
        failures.append(env.pa_state.total_failures)

    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_interruptions': np.mean(interruptions),
        'std_interruptions': np.std(interruptions),
        'mean_failures': np.mean(failures),
        'std_failures': np.std(failures),
        'rewards': rewards,
        'interruptions': interruptions,
        'failures': failures
    }


def train_ppo_policy(
    params: SimulationParams,
    task_def: TaskDefinition,
    total_timesteps: int = 100000,
    save_path: str = "ppo_assistant"
) -> PPO:
    """Train PPO policy for a specific task.

    Args:
        params: SimulationParams with memory/failure/cost parameters
        task_def: TaskDefinition specifying the procedural task
        total_timesteps: Number of training timesteps
        save_path: Path to save the trained model

    Returns:
        Trained PPO model
    """
    print(f"\n{'='*60}")
    print(f"Training PPO policy for: {task_def.task_name}")
    print(f"  n_steps={task_def.n_steps}, domain={task_def.domain}")
    print(f"  c_remind={params.c_remind}, c_fail={params.c_fail_base}, lambda={params.lambda_forget}")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"{'='*60}\n")

    # Create environment
    env = GymWrapperEnv(params, task_def)
    env = Monitor(env)

    # Create evaluation environment
    eval_env = GymWrapperEnv(params, task_def)
    eval_env = Monitor(eval_env)

    # Setup paths relative to project root
    tensorboard_path = str(PROJECT_ROOT / "models" / "tensorboard")
    model_save_path = str(PROJECT_ROOT / "models" / save_path.replace("ppo_assistant_", ""))

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=tensorboard_path
    )

    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_save_path,
        log_path=model_save_path,
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    # Train
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Save final model
    final_model_path = Path(model_save_path) / save_path / "final_model"
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(final_model_path))
    print(f"\nModel saved to {final_model_path}")

    return model


def compare_policies(
    params: SimulationParams,
    task_def: TaskDefinition,
    rl_model: PPO,
    n_episodes: int = 100
) -> Dict[str, Dict[str, Any]]:
    """Compare RL policy with baseline heuristics.

    Args:
        params: SimulationParams with memory/failure/cost parameters
        task_def: TaskDefinition specifying the procedural task
        rl_model: Trained PPO model
        n_episodes: Number of evaluation episodes

    Returns:
        Dictionary with results for each policy
    """
    print(f"\n{'='*60}")
    print(f"Comparing policies for {task_def.task_name} (n_episodes={n_episodes})")
    print(f"{'='*60}\n")

    env = ProcedureAssistantEnv(params, task_def)

    # Baseline policies (adapted to task size)
    policies = {
        'Random': RandomAssistantPolicy(task_def.n_steps),
        'Proactive': ProactiveReminderPolicy(task_def.n_steps, memory_threshold=0.3, lookahead=1),
        'Reactive': ReactivePolicyHighCost(task_def.n_steps, risk_threshold=0.25, params=params),
    }

    results = {}

    # Evaluate baseline policies
    for name, policy in policies.items():
        print(f"Evaluating {name} policy...")
        results[name] = evaluate_policy(env, policy, task_def, n_episodes)
        print(f"  Mean reward: {results[name]['mean_reward']:.2f} ± {results[name]['std_reward']:.2f}")
        print(f"  Mean interruptions: {results[name]['mean_interruptions']:.2f}")
        print(f"  Mean failures: {results[name]['mean_failures']:.2f}")
        print()

    # Evaluate RL policy
    print(f"Evaluating RL (PPO) policy...")
    results['RL_PPO'] = evaluate_policy(env, rl_model, task_def, n_episodes)
    print(f"  Mean reward: {results['RL_PPO']['mean_reward']:.2f} ± {results['RL_PPO']['std_reward']:.2f}")
    print(f"  Mean interruptions: {results['RL_PPO']['mean_interruptions']:.2f}")
    print(f"  Mean failures: {results['RL_PPO']['mean_failures']:.2f}")
    print()

    return results


def analyze_improvement(results: Dict[str, Dict[str, Any]]) -> None:
    """Analyze RL improvement over baselines"""
    print(f"\n{'='*60}")
    print("IMPROVEMENT ANALYSIS")
    print(f"{'='*60}\n")

    rl_reward = results['RL_PPO']['mean_reward']

    # Find best baseline
    baseline_names = ['Random', 'Proactive', 'Reactive']
    best_baseline = max(baseline_names, key=lambda x: results[x]['mean_reward'])
    best_baseline_reward = results[best_baseline]['mean_reward']

    print(f"Best baseline: {best_baseline}")
    print(f"  Reward: {best_baseline_reward:.2f}")
    print()

    print(f"RL (PPO):")
    print(f"  Reward: {rl_reward:.2f}")
    print()

    # Calculate improvement
    if best_baseline_reward < 0:
        # Both negative (costs): positive improvement = RL has higher (less negative) reward
        improvement = (rl_reward - best_baseline_reward) / abs(best_baseline_reward) * 100
        if rl_reward > best_baseline_reward:
            print(f"✓ RL IMPROVES by {improvement:.1f}%")
            print(f"  (Reduces cost from {abs(best_baseline_reward):.2f} to {abs(rl_reward):.2f})")
        else:
            print(f"✗ RL WORSE by {-improvement:.1f}%")
    else:
        improvement = (rl_reward - best_baseline_reward) / best_baseline_reward * 100
        print(f"Improvement: {improvement:+.1f}%")

    print()

    # Compare interruptions and failures
    print("Trade-off analysis:")
    for name in baseline_names + ['RL_PPO']:
        r = results[name]
        print(f"  {name:12s}: {r['mean_interruptions']:5.2f} interruptions, {r['mean_failures']:5.2f} failures")

    print()


def plot_comparison(results: Dict[str, Dict[str, Any]], params: SimulationParams, save_path: str = "rl_comparison.png"):
    """Plot comparison of policies"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    policy_names = ['Random', 'Proactive', 'Reactive', 'RL_PPO']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

    # Plot 1: Mean rewards
    ax = axes[0]
    rewards = [results[name]['mean_reward'] for name in policy_names]
    errors = [results[name]['std_reward'] for name in policy_names]
    bars = ax.bar(policy_names, rewards, yerr=errors, color=colors, alpha=0.7, capsize=5)
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('Policy Performance', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)

    # Highlight best
    best_idx = np.argmax(rewards)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)

    # Plot 2: Interruptions
    ax = axes[1]
    interruptions = [results[name]['mean_interruptions'] for name in policy_names]
    ax.bar(policy_names, interruptions, color=colors, alpha=0.7)
    ax.set_ylabel('Mean Interruptions', fontsize=12)
    ax.set_title('Interruption Frequency', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Failures
    ax = axes[2]
    failures = [results[name]['mean_failures'] for name in policy_names]
    ax.bar(policy_names, failures, color=colors, alpha=0.7)
    ax.set_ylabel('Mean Failures', fontsize=12)
    ax.set_title('Task Failures', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle(f'Policy Comparison (c_remind={params.c_remind}, c_fail={params.c_fail_base}, λ={params.lambda_forget})',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save to results/figures/
    figure_path = PROJECT_ROOT / "results" / "figures" / save_path
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(figure_path), dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to {figure_path}")
    plt.close()


def main():
    """Main training and comparison pipeline"""

    # Set random seed
    np.random.seed(42)

    # Load default task (make_cereal for testing, or specify via command line)
    import argparse
    parser = argparse.ArgumentParser(description='Train RL policy for procedural assistant')
    parser.add_argument('--task', type=str, default='make_cereal',
                       help='Task name (e.g., make_cereal, latte_making)')
    parser.add_argument('--timesteps', type=int, default=50000,
                       help='Total training timesteps')
    parser.add_argument('--regime', type=str, default='balanced',
                       help='Cost regime name')
    args = parser.parse_args()

    # Load task definition
    from task_definitions import get_task_definition
    task_def = get_task_definition(args.task)

    # Balanced cost regime
    params = SimulationParams(
        c_remind=5.0,
        c_fail_base=12.0,
        lambda_forget=0.05,
        f0_base=0.3,
        k_memory=2.0,
    )

    print("="*60)
    print("RL POLICY TRAINING AND EVALUATION")
    print("="*60)
    print(f"\nTask: {task_def.task_name} ({task_def.n_steps} steps, {task_def.domain})")
    print(f"Cost regime: c_remind={params.c_remind}, c_fail={params.c_fail_base}")
    print(f"Cost ratio: c_fail/c_remind = {params.c_fail_base/params.c_remind:.2f}")
    print(f"Forgetting rate: lambda={params.lambda_forget}")
    print()

    # Train RL policy
    save_path = f"ppo_assistant_{args.task}_{args.regime}"
    rl_model = train_ppo_policy(params, task_def, total_timesteps=args.timesteps, save_path=save_path)

    # Compare policies
    results = compare_policies(params, task_def, rl_model, n_episodes=100)

    # Analyze improvement
    analyze_improvement(results)

    # Plot comparison
    plot_comparison(results, params, save_path=f"rl_comparison_{args.task}_{args.regime}.png")

    # Save results
    results_serializable = {}
    for name, res in results.items():
        results_serializable[name] = {
            'mean_reward': float(res['mean_reward']),
            'std_reward': float(res['std_reward']),
            'mean_interruptions': float(res['mean_interruptions']),
            'std_interruptions': float(res['std_interruptions']),
            'mean_failures': float(res['mean_failures']),
            'std_failures': float(res['std_failures'])
        }

    results_path = PROJECT_ROOT / "data" / "results" / f"rl_results_{args.task}_{args.regime}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump({
            'task': args.task,
            'n_steps': task_def.n_steps,
            'domain': task_def.domain,
            'params': {
                'c_remind': params.c_remind,
                'c_fail_base': params.c_fail_base,
                'lambda_forget': params.lambda_forget,
                'f0_base': params.f0_base,
                'k_memory': params.k_memory
            },
            'results': results_serializable
        }, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
