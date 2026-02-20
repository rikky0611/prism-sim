"""
Analyze individual episodes to find successful intervention patterns

Look for episodes with:
- 4-5 interventions
- 0 failures
- Good reward
"""

import sys
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training"))

from procedure_assistant_sim import SimulationParams, ProcedureAssistantEnv
from task_definitions import get_task_definition, create_per_step_failure_costs
from train_rl_policy import GymWrapperEnv


def analyze_episodes(task_name='make_cereal', regime='balanced', n_episodes=100):
    """Analyze episodes looking for successful intervention patterns."""

    print(f"="*80)
    print(f"ANALYZING EPISODES: {task_name} - {regime}")
    print(f"="*80)

    # Load task and model
    task_def = get_task_definition(task_name)

    # Create environment with SIMPLIFIED memory model parameters
    regime_configs = {
        'very_high_stakes': {'c_fail': 30.0, 'f0_base': 0.3, 'lambda_forget': 0.05},
        'balanced': {'c_fail': 15.0, 'f0_base': 0.3, 'lambda_forget': 0.05},
        'moderate_low': {'c_fail': 10.0, 'f0_base': 0.3, 'lambda_forget': 0.05}
    }

    config = regime_configs[regime]
    c_fail_per_step = create_per_step_failure_costs(task_def, base_cost=config['c_fail'])

    params = SimulationParams(
        c_fail_per_step=c_fail_per_step,
        c_int=1.0,
        c_nar=0.0,
        c_resp=0.0,
        f0_base=config['f0_base'],
        lambda_forget=config['lambda_forget'],
        delta_reminder=0.6,  # NEW: Stronger boost
        k_memory=3.0,  # NEW: Steeper curve
        step_mean_duration=8,  # NEW: Tighter timing
        step_std_duration=2
    )

    gym_env = GymWrapperEnv(params, task_def)

    # Load model
    model_path = PROJECT_ROOT / "models" / regime / task_name / "final_model" / "final_model.zip"
    model = PPO.load(str(model_path))

    # Run episodes and collect data
    episodes_data = []

    for ep in range(n_episodes):
        obs, _ = gym_env.reset()
        done = False
        episode_reward = 0
        interruptions = 0
        actions_taken = []

        while not done:
            action = model.predict(obs, deterministic=True)[0]
            actions_taken.append(action)

            result = gym_env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result

            episode_reward += reward
            if action != 0:
                interruptions += 1

        failures = gym_env.env.pa_state.total_failures
        responses = gym_env.env.pa_state.total_responses  # NEW: Human responses
        narrations = gym_env.env.pa_state.total_narrations  # NEW: Human narrations

        episodes_data.append({
            'episode': ep,
            'reward': episode_reward,
            'interruptions': interruptions,
            'failures': failures,
            'responses': responses,  # NEW
            'narrations': narrations,  # NEW
            'actions': actions_taken
        })

    # Find successful patterns
    print(f"\nLooking for episodes with 4-5 interventions and 0 failures...")
    print("-"*80)

    successful = []
    for ep_data in episodes_data:
        if 4 <= ep_data['interruptions'] <= 5 and ep_data['failures'] == 0:
            successful.append(ep_data)

    print(f"Found {len(successful)} episodes with desired pattern (4-5 interventions, 0 failures)")

    if successful:
        print("\n" + "="*80)
        print("SUCCESS CASES:")
        print("="*80)
        for ep_data in successful[:10]:  # Show first 10
            print(f"\nEpisode {ep_data['episode']}:")
            print(f"  Reward: {ep_data['reward']:.1f}")
            print(f"    Breakdown:")
            print(f"      Interruptions: {ep_data['interruptions']} (cost: -{ep_data['interruptions'] * 1.0:.1f})")
            print(f"      Failures: {ep_data['failures']}")
            print(f"      Human responses: {ep_data['responses']} (cost: -{ep_data['responses'] * 2.0:.1f})")
            print(f"      Human narrations: {ep_data['narrations']} (cost: -{ep_data['narrations'] * 1.0:.1f})")
            print(f"  Actions: {ep_data['actions'][:20]}...")  # First 20 actions

    # Also show zero-failure cases regardless of interruptions
    print(f"\n" + "="*80)
    print("ALL ZERO-FAILURE EPISODES:")
    print("="*80)

    zero_failures = [ep for ep in episodes_data if ep['failures'] == 0]
    print(f"Found {len(zero_failures)} episodes with 0 failures")

    if zero_failures:
        # Show statistics
        interruption_counts = [ep['interruptions'] for ep in zero_failures]
        rewards = [ep['reward'] for ep in zero_failures]

        print(f"\nStatistics for zero-failure episodes:")
        print(f"  Mean interruptions: {np.mean(interruption_counts):.2f} ± {np.std(interruption_counts):.2f}")
        print(f"  Min interruptions: {min(interruption_counts)}")
        print(f"  Max interruptions: {max(interruption_counts)}")
        print(f"  Mean reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")

        print(f"\nBest examples (sorted by reward):")
        zero_failures_sorted = sorted(zero_failures, key=lambda x: x['reward'], reverse=True)
        for ep_data in zero_failures_sorted[:5]:
            int_cost = ep_data['interruptions'] * 1.0
            resp_cost = ep_data['responses'] * 2.0
            narr_cost = ep_data['narrations'] * 1.0
            print(f"  Episode {ep_data['episode']}: Reward={ep_data['reward']:.1f}")
            print(f"    Interruptions={ep_data['interruptions']} (-{int_cost:.1f}), Responses={ep_data['responses']} (-{resp_cost:.1f}), Narrations={ep_data['narrations']} (-{narr_cost:.1f})")

    # Overall statistics
    print(f"\n" + "="*80)
    print("OVERALL STATISTICS:")
    print("="*80)

    all_failures = [ep['failures'] for ep in episodes_data]
    all_interruptions = [ep['interruptions'] for ep in episodes_data]
    all_rewards = [ep['reward'] for ep in episodes_data]

    print(f"Mean failures: {np.mean(all_failures):.2f} ± {np.std(all_failures):.2f}")
    print(f"Mean interruptions: {np.mean(all_interruptions):.2f} ± {np.std(all_interruptions):.2f}")
    print(f"Mean reward: {np.mean(all_rewards):.1f} ± {np.std(all_rewards):.1f}")
    print(f"Zero-failure rate: {len(zero_failures)/len(episodes_data)*100:.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='make_cereal')
    parser.add_argument('--regime', type=str, default='balanced')
    parser.add_argument('--n-episodes', type=int, default=100)
    args = parser.parse_args()

    analyze_episodes(args.task, args.regime, args.n_episodes)
