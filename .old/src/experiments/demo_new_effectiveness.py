"""
Demonstration: How timing-dependent effectiveness should change RL behavior.

Compare old model (weak reminders) vs new model (strong, timing-dependent reminders)
on a simple baseline policy.
"""

import sys
from pathlib import Path
import numpy as np

# Add project paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))

from procedure_assistant_sim import (
    ProcedureAssistantEnv,
    SimulationParams,
    ProactiveReminderPolicy
)
from task_definitions import load_task_definitions


def run_demo():
    """Demonstrate difference between old and new models."""

    print("="*70)
    print("DEMONSTRATION: OLD MODEL vs NEW MODEL")
    print("="*70)
    print()

    # Load simple task
    tasks = load_task_definitions()
    task_def = tasks['make_cereal']  # 8-step task

    # Very high stakes scenario
    c_int = 2.0
    c_fail = 30.0
    print(f"Cost Structure: Very High Stakes")
    print(f"  c_int = {c_int} (interruption cost)")
    print(f"  c_fail = {c_fail} (failure cost)")
    print(f"  ratio = {c_fail/c_int:.1f}")
    print()

    # Create proactive policy (memory threshold = 0.3)
    policy = ProactiveReminderPolicy(task_def.n_steps, memory_threshold=0.3, lookahead=1)

    # Test 1: OLD MODEL (weak reminders)
    print("-"*70)
    print("TEST 1: OLD MODEL (weak 45% prevention)")
    print("-"*70)

    params_old = SimulationParams(
        f0_base=0.6,
        lambda_forget=0.10,
        lambda_recency=0.0,             # Disable recency
        effectiveness_recency=0.0,      # Disable recency
        c_int=c_int,
        c_fail_base=c_fail
    )
    params_old.apply_task_defaults(task_def)

    env_old = ProcedureAssistantEnv(params_old, task_def)
    obs_old = env_old.reset()

    total_reward_old = 0
    failures_old = 0
    interruptions_old = 0
    done = False

    while not done:
        action = policy.get_action(obs_old)
        obs_old, reward, done, info = env_old.step(action)
        total_reward_old += reward
        if info['failure']:
            failures_old += 1
        if action != 0:  # Not silent
            interruptions_old += 1

    print(f"Results:")
    print(f"  Total reward: {total_reward_old:.1f}")
    print(f"  Failures: {failures_old}")
    print(f"  Interruptions: {interruptions_old}")
    print()

    # Test 2: NEW MODEL (strong timing-dependent reminders)
    print("-"*70)
    print("TEST 2: NEW MODEL (strong 97% prevention when well-timed)")
    print("-"*70)

    params_new = SimulationParams(
        f0_base=0.6,
        lambda_forget=0.10,
        lambda_recency=0.20,            # Enable recency (fast decay)
        effectiveness_recency=0.95,     # Enable strong prevention
        c_int=c_int,
        c_fail_base=c_fail
    )
    params_new.apply_task_defaults(task_def)

    env_new = ProcedureAssistantEnv(params_new, task_def)
    obs_new = env_new.reset()

    total_reward_new = 0
    failures_new = 0
    interruptions_new = 0
    done = False

    while not done:
        action = policy.get_action(obs_new)
        obs_new, reward, done, info = env_new.step(action)
        total_reward_new += reward
        if info['failure']:
            failures_new += 1
        if action != 0:  # Not silent
            interruptions_new += 1

    print(f"Results:")
    print(f"  Total reward: {total_reward_new:.1f}")
    print(f"  Failures: {failures_new}")
    print(f"  Interruptions: {interruptions_new}")
    print()

    # Comparison
    print("="*70)
    print("COMPARISON")
    print("="*70)
    print(f"Reward improvement: {total_reward_new - total_reward_old:+.1f}")
    print(f"Failure reduction: {failures_old - failures_new:+d}")
    print(f"Interruption change: {interruptions_new - interruptions_old:+d}")
    print()
    print("INSIGHT:")
    if total_reward_new > total_reward_old:
        print("  With effective reminders (90-100%), interruptions become MORE valuable!")
        print("  This should REVERSE the counter-intuitive pattern:")
        print("  - High stakes → MORE interventions (because they actually prevent failures)")
        print("  - Low stakes → FEWER interventions (because cost/benefit less favorable)")
    else:
        print("  Results may vary due to stochasticity. Run multiple episodes for stability.")
    print("="*70)


if __name__ == '__main__':
    # Set seed for reproducibility
    np.random.seed(42)
    run_demo()
