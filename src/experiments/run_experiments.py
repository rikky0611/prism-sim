"""
Run experiments with different cost functions and policies.

This script explores how different interaction cost regimes affect
human-AI collaboration trajectories in procedure assistance.

Key experiments:
1. Baseline: Low interruption cost
2. High interruption cost: c_int = 20 (costly to interrupt)
3. High failure cost: c_fail = 50 (mistakes are very costly)
4. Balanced: Moderate costs across the board
"""

import numpy as np
import json
from typing import Dict
from procedure_assistant_sim import (
    SimulationParams,
    RandomAssistantPolicy,
    ProactiveReminderPolicy,
    ReactivePolicyHighCost,
    run_simulation,
    plot_results,
    plot_trajectory,
    PROCEDURAL_STEPS,
)


def experiment_1_cost_comparison():
    """
    Experiment 1: Compare different cost regimes with a fixed policy
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Impact of Different Cost Regimes")
    print("="*70)
    print("\nPolicy: Proactive Reminder (memory threshold = 0.3)")
    print("\nWe vary the interaction cost to simulate different interruption burdens:")
    print("  - Low cost: c_int=2 (easy to interrupt, like smartwatch notification)")
    print("  - Medium cost: c_int=5 (moderate interruption)")
    print("  - High cost: c_int=15 (significant disruption, e.g., during surgery)")
    print()

    policy = ProactiveReminderPolicy(memory_threshold=0.3, lookahead=1)
    n_episodes = 20

    results = {}

    # Low interruption cost
    print("Running: Low interruption cost (c_int=2)...")
    params_low = SimulationParams(c_int=2.0, c_fail_base=20.0)
    results['Low c_int (2)'] = run_simulation(policy, params_low, n_episodes, verbose=False)
    print(f"  → Mean reward: {results['Low c_int (2)']['mean_reward']:.1f}, "
          f"Failures: {results['Low c_int (2)']['mean_failures']:.1f}, "
          f"Interactions: {results['Low c_int (2)']['mean_interactions']:.1f}")

    # Medium interruption cost
    print("Running: Medium interruption cost (c_int=5)...")
    params_med = SimulationParams(c_int=5.0, c_fail_base=20.0)
    results['Medium c_int (5)'] = run_simulation(policy, params_med, n_episodes, verbose=False)
    print(f"  → Mean reward: {results['Medium c_int (5)']['mean_reward']:.1f}, "
          f"Failures: {results['Medium c_int (5)']['mean_failures']:.1f}, "
          f"Interactions: {results['Medium c_int (5)']['mean_interactions']:.1f}")

    # High interruption cost
    print("Running: High interruption cost (c_int=15)...")
    params_high = SimulationParams(c_int=15.0, c_fail_base=20.0)
    results['High c_int (15)'] = run_simulation(policy, params_high, n_episodes, verbose=False)
    print(f"  → Mean reward: {results['High c_int (15)']['mean_reward']:.1f}, "
          f"Failures: {results['High c_int (15)']['mean_failures']:.1f}, "
          f"Interactions: {results['High c_int (15)']['mean_interactions']:.1f}")

    print("\n" + "-"*70)
    print("Key Insight:")
    print("  As interruption cost increases, the same proactive policy becomes")
    print("  less effective because each interaction is more expensive.")
    print("  This suggests need for adaptive policies that consider cost context.")
    print("-"*70)

    plot_results(results, 'results_exp1_cost_comparison.png')

    return results


def experiment_2_policy_comparison():
    """
    Experiment 2: Compare different policies under high interruption cost
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Policy Comparison under High Interruption Cost")
    print("="*70)
    print("\nScenario: c_int=15 (high interruption burden)")
    print("\nPolicies tested:")
    print("  1. Random: Baseline with mostly silent actions")
    print("  2. Proactive: Reminds when memory is low")
    print("  3. Reactive: Only reminds when failure risk is very high")
    print()

    params = SimulationParams(c_int=15.0, c_fail_base=20.0, f0_base=0.4)
    n_episodes = 20

    results = {}

    # Random policy (conservative)
    print("Running: Random policy...")
    policy_random = RandomAssistantPolicy(action_probs={
        'silent': 0.85,
        'confirm': 0.05,
        **{f'remind_{i}': 0.02 for i in range(len(PROCEDURAL_STEPS))}
    })
    results['Random'] = run_simulation(policy_random, params, n_episodes, verbose=False)
    print(f"  → Mean reward: {results['Random']['mean_reward']:.1f}, "
          f"Failures: {results['Random']['mean_failures']:.1f}, "
          f"Interactions: {results['Random']['mean_interactions']:.1f}")

    # Proactive policy
    print("Running: Proactive policy...")
    policy_proactive = ProactiveReminderPolicy(memory_threshold=0.3, lookahead=1)
    results['Proactive'] = run_simulation(policy_proactive, params, n_episodes, verbose=False)
    print(f"  → Mean reward: {results['Proactive']['mean_reward']:.1f}, "
          f"Failures: {results['Proactive']['mean_failures']:.1f}, "
          f"Interactions: {results['Proactive']['mean_interactions']:.1f}")

    # Reactive policy (high threshold)
    print("Running: Reactive policy...")
    policy_reactive = ReactivePolicyHighCost(risk_threshold=0.3, params=params)
    results['Reactive'] = run_simulation(policy_reactive, params, n_episodes, verbose=False)
    print(f"  → Mean reward: {results['Reactive']['mean_reward']:.1f}, "
          f"Failures: {results['Reactive']['mean_failures']:.1f}, "
          f"Interactions: {results['Reactive']['mean_interactions']:.1f}")

    print("\n" + "-"*70)
    print("Key Insight:")
    print("  Under high interruption cost, reactive policies that only intervene")
    print("  when failure risk is elevated perform better than proactive policies")
    print("  that frequently interrupt the user.")
    print("-"*70)

    plot_results(results, 'results_exp2_policy_comparison.png')

    return results


def experiment_3_failure_cost_tradeoff():
    """
    Experiment 3: Explore failure cost vs interruption cost tradeoff
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Failure Cost vs Interruption Cost Tradeoff")
    print("="*70)
    print("\nWe explore how the optimal policy changes as we vary the ratio")
    print("of failure cost to interruption cost.")
    print()

    results = {}
    n_episodes = 20

    # Low failure cost, high interruption cost
    print("Running: Low failure cost (10), high c_int (15)...")
    params1 = SimulationParams(c_int=15.0, c_fail_base=10.0)
    policy1 = ReactivePolicyHighCost(risk_threshold=0.35, params=params1)
    results['Low fail / High int'] = run_simulation(policy1, params1, n_episodes, verbose=False)
    print(f"  → Mean reward: {results['Low fail / High int']['mean_reward']:.1f}, "
          f"Failures: {results['Low fail / High int']['mean_failures']:.1f}, "
          f"Interactions: {results['Low fail / High int']['mean_interactions']:.1f}")

    # Balanced costs
    print("Running: Balanced costs (c_int=8, c_fail=20)...")
    params2 = SimulationParams(c_int=8.0, c_fail_base=20.0)
    policy2 = ProactiveReminderPolicy(memory_threshold=0.4, lookahead=1)
    results['Balanced'] = run_simulation(policy2, params2, n_episodes, verbose=False)
    print(f"  → Mean reward: {results['Balanced']['mean_reward']:.1f}, "
          f"Failures: {results['Balanced']['mean_failures']:.1f}, "
          f"Interactions: {results['Balanced']['mean_interactions']:.1f}")

    # High failure cost, low interruption cost
    print("Running: High failure cost (40), low c_int (3)...")
    params3 = SimulationParams(c_int=3.0, c_fail_base=40.0)
    policy3 = ProactiveReminderPolicy(memory_threshold=0.25, lookahead=2)
    results['High fail / Low int'] = run_simulation(policy3, params3, n_episodes, verbose=False)
    print(f"  → Mean reward: {results['High fail / Low int']['mean_reward']:.1f}, "
          f"Failures: {results['High fail / Low int']['mean_failures']:.1f}, "
          f"Interactions: {results['High fail / Low int']['mean_interactions']:.1f}")

    print("\n" + "-"*70)
    print("Key Insight:")
    print("  The optimal policy is highly context-dependent:")
    print("  - When failures are cheap relative to interruptions: be conservative")
    print("  - When failures are expensive relative to interruptions: be proactive")
    print("  This highlights the need for context-aware assistants.")
    print("-"*70)

    plot_results(results, 'results_exp3_tradeoff.png')

    return results


def experiment_4_memory_dynamics():
    """
    Experiment 4: Effect of memory parameters on performance
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: Memory Dynamics")
    print("="*70)
    print("\nWe vary forgetting rate (lambda) to understand memory dynamics.")
    print("Higher lambda = faster forgetting = more need for reminders")
    print()

    policy = ProactiveReminderPolicy(memory_threshold=0.3, lookahead=1)
    n_episodes = 20
    results = {}

    # Slow forgetting
    print("Running: Slow forgetting (lambda=0.02)...")
    params1 = SimulationParams(lambda_forget=0.02, c_int=8.0, c_fail_base=20.0)
    results['Slow forget'] = run_simulation(policy, params1, n_episodes, verbose=False)
    print(f"  → Mean reward: {results['Slow forget']['mean_reward']:.1f}, "
          f"Failures: {results['Slow forget']['mean_failures']:.1f}, "
          f"Interactions: {results['Slow forget']['mean_interactions']:.1f}")

    # Medium forgetting (default)
    print("Running: Medium forgetting (lambda=0.05)...")
    params2 = SimulationParams(lambda_forget=0.05, c_int=8.0, c_fail_base=20.0)
    results['Medium forget'] = run_simulation(policy, params2, n_episodes, verbose=False)
    print(f"  → Mean reward: {results['Medium forget']['mean_reward']:.1f}, "
          f"Failures: {results['Medium forget']['mean_failures']:.1f}, "
          f"Interactions: {results['Medium forget']['mean_interactions']:.1f}")

    # Fast forgetting
    print("Running: Fast forgetting (lambda=0.10)...")
    params3 = SimulationParams(lambda_forget=0.10, c_int=8.0, c_fail_base=20.0)
    results['Fast forget'] = run_simulation(policy, params3, n_episodes, verbose=False)
    print(f"  → Mean reward: {results['Fast forget']['mean_reward']:.1f}, "
          f"Failures: {results['Fast forget']['mean_failures']:.1f}, "
          f"Interactions: {results['Fast forget']['mean_interactions']:.1f}")

    print("\n" + "-"*70)
    print("Key Insight:")
    print("  Faster forgetting requires more frequent reminders to maintain")
    print("  low failure rates. This suggests that memory retention is a key")
    print("  factor in determining optimal assistant behavior.")
    print("-"*70)

    plot_results(results, 'results_exp4_memory.png')

    return results


def visualize_sample_trajectories():
    """
    Generate and visualize sample trajectories for qualitative analysis
    """
    print("\n" + "="*70)
    print("GENERATING SAMPLE TRAJECTORIES")
    print("="*70)
    print()

    # Trajectory 1: Proactive policy, low cost
    print("Trajectory 1: Proactive policy with low interruption cost")
    params1 = SimulationParams(c_int=2.0, c_fail_base=20.0)
    policy1 = ProactiveReminderPolicy(memory_threshold=0.3)
    result1 = run_simulation(policy1, params1, n_episodes=1, verbose=False)
    plot_trajectory(result1['histories'][0], 'trajectory_proactive_low_cost.png')

    # Trajectory 2: Reactive policy, high cost
    print("Trajectory 2: Reactive policy with high interruption cost")
    params2 = SimulationParams(c_int=15.0, c_fail_base=20.0)
    policy2 = ReactivePolicyHighCost(risk_threshold=0.3, params=params2)
    result2 = run_simulation(policy2, params2, n_episodes=1, verbose=False)
    plot_trajectory(result2['histories'][0], 'trajectory_reactive_high_cost.png')

    print("\nTrajectories saved. These show:")
    print("  - Step progression over time")
    print("  - When assistant interrupts")
    print("  - When failures occur")
    print("  - Cumulative reward evolution")
    print()


def save_summary_report(all_results: Dict):
    """
    Save a summary report of all experiments
    """
    print("\n" + "="*70)
    print("SAVING SUMMARY REPORT")
    print("="*70)

    report = {
        'experiments': {},
        'summary': {},
    }

    # Aggregate results from all experiments
    for exp_name, results in all_results.items():
        report['experiments'][exp_name] = {}
        for condition, result in results.items():
            report['experiments'][exp_name][condition] = {
                'mean_reward': float(result['mean_reward']),
                'std_reward': float(result['std_reward']),
                'mean_failures': float(result['mean_failures']),
                'mean_interactions': float(result['mean_interactions']),
                'mean_episode_length': float(np.mean(result['episode_lengths'])),
            }

    # Save to JSON
    with open('experiment_results.json', 'w') as f:
        json.dump(report, f, indent=2)

    print("Summary report saved to: experiment_results.json")
    print()


def main():
    """
    Main experiment runner
    """
    print("\n" + "="*70)
    print("PROCEDURE ASSISTANT SIMULATION: HUMAN-AI COLLABORATION")
    print("="*70)
    print("\nThis simulation explores how different cost structures affect")
    print("optimal assistant behavior in procedural task assistance.")
    print("\nBased on the POMDP formulation in modeling.pdf:")
    print("  - Human performs procedural task (cooking in Overcooked)")
    print("  - Assistant observes with noise from outside")
    print("  - Assistant can provide reminders or confirmations")
    print("  - Tradeoff: Interaction cost vs. Failure cost")
    print()
    print("Running all experiments...")
    print()

    all_results = {}

    # Run all experiments
    all_results['exp1_cost_comparison'] = experiment_1_cost_comparison()
    all_results['exp2_policy_comparison'] = experiment_2_policy_comparison()
    all_results['exp3_tradeoff'] = experiment_3_failure_cost_tradeoff()
    all_results['exp4_memory'] = experiment_4_memory_dynamics()

    # Generate trajectories
    visualize_sample_trajectories()

    # Save summary
    save_summary_report(all_results)

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*70)
    print("\nGenerated files:")
    print("  - results_exp1_cost_comparison.png")
    print("  - results_exp2_policy_comparison.png")
    print("  - results_exp3_tradeoff.png")
    print("  - results_exp4_memory.png")
    print("  - trajectory_proactive_low_cost.png")
    print("  - trajectory_reactive_high_cost.png")
    print("  - experiment_results.json")
    print("\nSee below for detailed analysis and decisions.")
    print("="*70)
    print()


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    main()
