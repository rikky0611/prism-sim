"""
Train RL policies on multiple cost regimes to explore intervention strategies
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))

from train_rl_policy import (
    GymWrapperEnv,
    evaluate_policy,
)
from procedure_assistant_sim import (
    SimulationParams,
    ProcedureAssistantEnv,
    RandomAssistantPolicy,
    ProactiveReminderPolicy,
    ReactivePolicyHighCost
)


def train_on_regime(regime_name, c_int, c_fail, timesteps=50000):
    """Train PPO on a specific cost regime"""
    print(f"\n{'='*70}")
    print(f"Training on {regime_name}")
    print(f"  c_int={c_int}, c_fail={c_fail}, ratio={c_fail/c_int:.2f}")
    print(f"{'='*70}\n")

    # Set seed for reproducibility
    np.random.seed(42)

    # Create parameters
    params = SimulationParams(
        c_int=c_int,
        c_fail_base=c_fail,
        lambda_forget=0.05,
        f0_base=0.3,
        k_memory=2.0,
    )

    # Create and train
    env = GymWrapperEnv(params)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    print("Training...")
    model.learn(total_timesteps=timesteps, progress_bar=True)

    # Save model to models/multi_regime_v1/
    model_dir = PROJECT_ROOT / "models" / "multi_regime_v1" / f"ppo_assistant_{regime_name}"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir / "final_model"))
    print(f"Model saved to {model_dir}/")

    return model, params


def evaluate_all_policies(params, rl_model, n_episodes=100):
    """Evaluate all policies on given parameters"""
    env = ProcedureAssistantEnv(params)

    policies = {
        'Random': RandomAssistantPolicy(),
        'Proactive': ProactiveReminderPolicy(memory_threshold=0.3, lookahead=1),
        'Reactive': ReactivePolicyHighCost(risk_threshold=0.25, params=params),
        'RL_PPO': rl_model,
    }

    results = {}
    for name, policy in policies.items():
        results[name] = evaluate_policy(env, policy, n_episodes, seed=42)

    return results


def print_regime_results(regime_name, c_int, c_fail, results):
    """Print results for a regime"""
    print(f"\n{'='*70}")
    print(f"RESULTS: {regime_name} (c_int={c_int}, c_fail={c_fail}, ratio={c_fail/c_int:.2f})")
    print(f"{'='*70}\n")

    # Print table
    print(f"{'Policy':<15} {'Reward':<12} {'Interruptions':<15} {'Failures':<10} {'Interv/Episode'}")
    print("-" * 70)

    for name in ['Random', 'Proactive', 'Reactive', 'RL_PPO']:
        r = results[name]
        # Calculate actual intervention rate (interruptions that aren't silent)
        interv_rate = r['mean_interruptions']
        print(f"{name:<15} {r['mean_reward']:>8.2f}     "
              f"{r['mean_interruptions']:>8.2f}        "
              f"{r['mean_failures']:>8.2f}     "
              f"{interv_rate:>6.2f}")

    # Find best
    best = max(results.keys(), key=lambda k: results[k]['mean_reward'])
    rl_reward = results['RL_PPO']['mean_reward']
    best_baseline_reward = max(results[k]['mean_reward'] for k in results if k != 'RL_PPO')

    if rl_reward > best_baseline_reward:
        improvement = (rl_reward - best_baseline_reward) / abs(best_baseline_reward) * 100
        print(f"\n✓ RL is BEST with {improvement:+.1f}% improvement over best baseline")
    else:
        print(f"\n✗ Best baseline ({best}) performs better")

    # Analyze RL behavior
    rl_int = results['RL_PPO']['mean_interruptions']
    rl_fail = results['RL_PPO']['mean_failures']

    print(f"\nRL Strategy Analysis:")
    if rl_int < 1:
        print(f"  → Nearly silent ({rl_int:.2f} interventions/episode)")
    elif rl_int < 5:
        print(f"  → Conservative ({rl_int:.2f} interventions/episode)")
    elif rl_int < 10:
        print(f"  → Moderate ({rl_int:.2f} interventions/episode)")
    else:
        print(f"  → Active ({rl_int:.2f} interventions/episode)")

    print(f"  → Accepts {rl_fail:.2f} failures/episode")


def plot_multi_regime_comparison(all_results, regimes):
    """Create comprehensive comparison plot"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    regime_names = [r[0] for r in regimes]
    n_regimes = len(regimes)

    policy_names = ['Random', 'Proactive', 'Reactive', 'RL_PPO']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

    # Plot 1: Rewards by regime
    ax = axes[0, 0]
    x = np.arange(n_regimes)
    width = 0.2
    for i, policy in enumerate(policy_names):
        rewards = [all_results[regime][policy]['mean_reward'] for regime in regime_names]
        ax.bar(x + i*width, rewards, width, label=policy, color=colors[i], alpha=0.8)
    ax.set_xlabel('Cost Regime', fontsize=12)
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('Performance Across Cost Regimes', fontsize=14, fontweight='bold')
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels(regime_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

    # Plot 2: Interruptions by regime
    ax = axes[0, 1]
    for i, policy in enumerate(policy_names):
        interruptions = [all_results[regime][policy]['mean_interruptions'] for regime in regime_names]
        ax.bar(x + i*width, interruptions, width, label=policy, color=colors[i], alpha=0.8)
    ax.set_xlabel('Cost Regime', fontsize=12)
    ax.set_ylabel('Mean Interruptions', fontsize=12)
    ax.set_title('Intervention Frequency', fontsize=14, fontweight='bold')
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels(regime_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Failures by regime
    ax = axes[0, 2]
    for i, policy in enumerate(policy_names):
        failures = [all_results[regime][policy]['mean_failures'] for regime in regime_names]
        ax.bar(x + i*width, failures, width, label=policy, color=colors[i], alpha=0.8)
    ax.set_xlabel('Cost Regime', fontsize=12)
    ax.set_ylabel('Mean Failures', fontsize=12)
    ax.set_title('Task Failures', fontsize=14, fontweight='bold')
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels(regime_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 4: RL improvement over best baseline
    ax = axes[1, 0]
    improvements = []
    for regime_name in regime_names:
        rl_reward = all_results[regime_name]['RL_PPO']['mean_reward']
        baseline_rewards = [all_results[regime_name][p]['mean_reward']
                          for p in policy_names if p != 'RL_PPO']
        best_baseline = max(baseline_rewards)
        improvement = (rl_reward - best_baseline) / abs(best_baseline) * 100
        improvements.append(improvement)

    bars = ax.bar(regime_names, improvements, color=['green' if i > 0 else 'red' for i in improvements], alpha=0.7)
    ax.set_xlabel('Cost Regime', fontsize=12)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('RL Improvement Over Best Baseline', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.1f}%', ha='center', va='bottom' if val > 0 else 'top', fontsize=10)

    # Plot 5: Cost ratio vs RL intervention rate
    ax = axes[1, 1]
    cost_ratios = [r[2]/r[1] for r in regimes]
    rl_interventions = [all_results[regime]['RL_PPO']['mean_interruptions'] for regime in regime_names]
    ax.scatter(cost_ratios, rl_interventions, s=200, alpha=0.6, color='#FFA07A', edgecolors='black', linewidth=2)

    for i, regime in enumerate(regime_names):
        ax.annotate(regime, (cost_ratios[i], rl_interventions[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax.set_xlabel('Cost Ratio (c_fail / c_int)', fontsize=12)
    ax.set_ylabel('RL Interventions/Episode', fontsize=12)
    ax.set_title('RL Intervention Strategy vs Cost Ratio', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(cost_ratios, rl_interventions, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(cost_ratios), max(cost_ratios), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    ax.legend()

    # Plot 6: Trade-off analysis for RL
    ax = axes[1, 2]
    rl_interruptions = [all_results[regime]['RL_PPO']['mean_interruptions'] for regime in regime_names]
    rl_failures = [all_results[regime]['RL_PPO']['mean_failures'] for regime in regime_names]

    scatter = ax.scatter(rl_interruptions, rl_failures, s=200, c=cost_ratios,
                        cmap='viridis', alpha=0.6, edgecolors='black', linewidth=2)

    for i, regime in enumerate(regime_names):
        ax.annotate(regime, (rl_interruptions[i], rl_failures[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax.set_xlabel('Interruptions/Episode', fontsize=12)
    ax.set_ylabel('Failures/Episode', fontsize=12)
    ax.set_title('RL Trade-off: Interruptions vs Failures', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cost Ratio (c_fail/c_int)', fontsize=10)

    plt.tight_layout()
    figure_path = PROJECT_ROOT / "results" / "figures" / "rl_multi_regime_comparison.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(figure_path), dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to {figure_path}")
    plt.close()


def main():
    """Train and evaluate on multiple cost regimes"""

    # Define cost regimes to explore
    regimes = [
        # (name, c_int, c_fail, description)
        ("high_stakes", 2, 20, "Surgery/critical - expect proactive"),
        ("moderate", 3, 15, "Important task - expect selective"),
        ("balanced", 5, 12, "Balanced - baseline test"),
        ("low_stakes", 5, 8, "Casual task - expect some intervention"),
        ("high_cost_int", 10, 12, "Deep work - expect minimal"),
    ]

    print("="*70)
    print("MULTI-REGIME RL TRAINING AND EVALUATION")
    print("="*70)
    print(f"\nTraining on {len(regimes)} cost regimes:")
    for name, c_int, c_fail, desc in regimes:
        print(f"  {name:15s}: c_int={c_int:2d}, c_fail={c_fail:2d}, ratio={c_fail/c_int:4.1f} - {desc}")
    print()

    all_results = {}
    all_models = {}

    # Train on each regime
    for regime_name, c_int, c_fail, desc in regimes:
        model, params = train_on_regime(regime_name, c_int, c_fail, timesteps=50000)
        results = evaluate_all_policies(params, model, n_episodes=100)

        all_results[regime_name] = results
        all_models[regime_name] = model

        print_regime_results(regime_name, c_int, c_fail, results)

    # Create comprehensive comparison plot
    plot_multi_regime_comparison(all_results, regimes)

    # Save all results
    results_serializable = {}
    for regime_name in all_results:
        results_serializable[regime_name] = {}
        for policy_name, res in all_results[regime_name].items():
            results_serializable[regime_name][policy_name] = {
                'mean_reward': float(res['mean_reward']),
                'std_reward': float(res['std_reward']),
                'mean_interruptions': float(res['mean_interruptions']),
                'mean_failures': float(res['mean_failures']),
            }

    # Add regime parameters
    results_serializable['regimes'] = {
        name: {'c_int': c_int, 'c_fail': c_fail, 'description': desc}
        for name, c_int, c_fail, desc in regimes
    }

    results_path = PROJECT_ROOT / "data" / "results" / "rl_multi_regime_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print("\n" + "="*70)
    print("MULTI-REGIME ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to:")
    print(f"  - {results_path}")
    print(f"  - {PROJECT_ROOT / 'results' / 'figures' / 'rl_multi_regime_comparison.png'}")
    print(f"  - {PROJECT_ROOT / 'models' / 'multi_regime_v1'} (trained models)")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    for regime_name, c_int, c_fail, desc in regimes:
        rl_int = all_results[regime_name]['RL_PPO']['mean_interruptions']
        rl_fail = all_results[regime_name]['RL_PPO']['mean_failures']
        print(f"{regime_name:15s} (ratio={c_fail/c_int:4.1f}): "
              f"{rl_int:5.2f} interventions, {rl_fail:5.2f} failures")


if __name__ == "__main__":
    main()
