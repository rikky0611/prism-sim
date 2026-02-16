"""
Train RL policies on multiple cost regimes with HIGHER FAILURE RISK
to encourage more interesting intervention behaviors
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


def train_on_regime_v2(regime_name, c_int, c_fail, f0_base=0.6, lambda_forget=0.10, timesteps=50000):
    """
    Train PPO on a specific cost regime with HIGHER failure risk

    Key changes from v1:
    - f0_base increased from 0.3 to 0.6 (higher base failure probability)
    - lambda_forget increased from 0.05 to 0.10 (faster memory decay)
    - This makes reminders more valuable!
    """
    print(f"\n{'='*70}")
    print(f"Training on {regime_name}")
    print(f"  c_int={c_int}, c_fail={c_fail}, ratio={c_fail/c_int:.2f}")
    print(f"  f0={f0_base}, λ={lambda_forget} (HIGH RISK PARAMETERS)")
    print(f"{'='*70}\n")

    np.random.seed(42)

    params = SimulationParams(
        c_int=c_int,
        c_fail_base=c_fail,
        lambda_forget=lambda_forget,  # Faster forgetting!
        f0_base=f0_base,              # Higher base failure!
        k_memory=2.0,
    )

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

    # Save to models/multi_regime_v2/
    model_dir = PROJECT_ROOT / "models" / "multi_regime_v2" / f"ppo_assistant_v2_{regime_name}"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir / "final_model"))
    print(f"Model saved to {model_dir}/")

    return model, params


def evaluate_all_policies_v2(params, rl_model, n_episodes=100):
    """Evaluate all policies"""
    env = ProcedureAssistantEnv(params)

    policies = {
        'Random': RandomAssistantPolicy(),
        'Proactive': ProactiveReminderPolicy(memory_threshold=0.3, lookahead=1),
        'Reactive': ReactivePolicyHighCost(risk_threshold=0.4, params=params),  # Higher threshold for high-risk env
        'RL_PPO': rl_model,
    }

    results = {}
    for name, policy in policies.items():
        results[name] = evaluate_policy(env, policy, n_episodes, seed=42)

    return results


def print_regime_results_v2(regime_name, c_int, c_fail, results):
    """Print results for a regime"""
    print(f"\n{'='*70}")
    print(f"RESULTS: {regime_name} (c_int={c_int}, c_fail={c_fail}, ratio={c_fail/c_int:.2f})")
    print(f"{'='*70}\n")

    print(f"{'Policy':<15} {'Reward':<12} {'Interruptions':<15} {'Failures':<10}")
    print("-" * 60)

    for name in ['Random', 'Proactive', 'Reactive', 'RL_PPO']:
        r = results[name]
        print(f"{name:<15} {r['mean_reward']:>8.2f}     "
              f"{r['mean_interruptions']:>8.2f}        "
              f"{r['mean_failures']:>8.2f}")

    best_baseline = max((results[k]['mean_reward'] for k in results if k != 'RL_PPO'))
    rl_reward = results['RL_PPO']['mean_reward']

    if rl_reward > best_baseline:
        improvement = (rl_reward - best_baseline) / abs(best_baseline) * 100
        print(f"\n✓ RL is BEST with {improvement:+.1f}% improvement")
    else:
        diff = (rl_reward - best_baseline) / abs(best_baseline) * 100
        print(f"\n✗ RL underperforms by {diff:.1f}%")

    rl_int = results['RL_PPO']['mean_interruptions']
    rl_fail = results['RL_PPO']['mean_failures']

    print(f"\nRL Learned Strategy:")
    if rl_int < 1:
        print(f"  ⚠️  Nearly silent ({rl_int:.2f} interventions) - NOT INTERESTING")
    elif rl_int < 5:
        print(f"  ✓ Conservative intervention ({rl_int:.2f} interventions) - INTERESTING!")
    elif rl_int < 10:
        print(f"  ✓ Moderate intervention ({rl_int:.2f} interventions) - INTERESTING!")
    else:
        print(f"  ✓ Active intervention ({rl_int:.2f} interventions) - INTERESTING!")

    print(f"  → Failures: {rl_fail:.2f}/episode")

    # Compare with baselines
    print(f"\n  Compared to Proactive: {results['Proactive']['mean_interruptions']:.2f} int, {results['Proactive']['mean_failures']:.2f} fail")
    print(f"  Compared to Reactive:  {results['Reactive']['mean_interruptions']:.2f} int, {results['Reactive']['mean_failures']:.2f} fail")


def main():
    """Train and evaluate on multiple cost regimes with HIGH RISK parameters"""

    regimes = [
        # (name, c_int, c_fail, description)
        ("very_high_stakes", 2, 30, "Surgery - expect many interventions"),
        ("high_stakes", 2, 20, "Critical task - expect proactive"),
        ("moderate_high", 3, 15, "Important task - expect selective"),
        ("balanced", 5, 15, "Balanced - expect some intervention"),
        ("moderate_low", 5, 10, "Casual task - expect minimal"),
    ]

    print("="*70)
    print("MULTI-REGIME RL TRAINING V2 - HIGH FAILURE RISK")
    print("="*70)
    print(f"\nEnvironment: f0=0.6 (was 0.3), λ=0.10 (was 0.05)")
    print(f"Goal: Make reminders more valuable to encourage intervention\n")
    print(f"Training on {len(regimes)} cost regimes:")
    for name, c_int, c_fail, desc in regimes:
        print(f"  {name:20s}: c_int={c_int:2d}, c_fail={c_fail:2d}, ratio={c_fail/c_int:4.1f} - {desc}")
    print()

    all_results = {}
    all_models = {}

    for regime_name, c_int, c_fail, desc in regimes:
        model, params = train_on_regime_v2(
            regime_name, c_int, c_fail,
            f0_base=0.6,        # Higher base failure!
            lambda_forget=0.10,  # Faster forgetting!
            timesteps=50000
        )
        results = evaluate_all_policies_v2(params, model, n_episodes=100)

        all_results[regime_name] = results
        all_models[regime_name] = model

        print_regime_results_v2(regime_name, c_int, c_fail, results)

    # Save results
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

    results_serializable['config'] = {
        'f0_base': 0.6,
        'lambda_forget': 0.10,
        'k_memory': 2.0,
    }

    results_serializable['regimes'] = {
        name: {'c_int': c_int, 'c_fail': c_fail, 'description': desc}
        for name, c_int, c_fail, desc in regimes
    }

    results_path = PROJECT_ROOT / "data" / "results" / "rl_multi_regime_v2_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY - INTERESTING BEHAVIORS?")
    print(f"{'='*70}\n")

    print(f"{'Regime':<20s} {'Ratio':>6s} {'RL_Int':>8s} {'RL_Fail':>8s} {'Interesting?'}")
    print("-" * 70)

    interesting_count = 0
    for regime_name, c_int, c_fail, desc in regimes:
        rl_int = all_results[regime_name]['RL_PPO']['mean_interruptions']
        rl_fail = all_results[regime_name]['RL_PPO']['mean_failures']

        interesting = "✓ YES" if rl_int >= 1.0 else "✗ No (too silent)"
        if rl_int >= 1.0:
            interesting_count += 1

        print(f"{regime_name:<20s} {c_fail/c_int:>6.1f} {rl_int:>8.2f} {rl_fail:>8.2f}  {interesting}")

    print(f"\nInteresting behaviors found: {interesting_count}/{len(regimes)}")

    if interesting_count >= 3:
        print("✓ SUCCESS: Multiple regimes show nuanced intervention strategies!")
    elif interesting_count >= 1:
        print("⚠️  PARTIAL: Some interesting behaviors, but still too much silence")
    else:
        print("✗ FAILED: RL still learns to stay silent everywhere")

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
