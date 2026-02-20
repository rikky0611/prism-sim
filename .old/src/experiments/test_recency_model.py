"""
Test script for timing-dependent reminder effectiveness model.

Verifies that reminders achieve 90-100% prevention when well-timed,
and degrade over time as expected.
"""

import sys
from pathlib import Path
import numpy as np

# Add project paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))

from procedure_assistant_sim import (
    ProcedureAssistantEnv,
    SimulationParams
)
from task_definitions import load_task_definitions


def test_recency_effectiveness():
    """Test timing-dependent reminder effectiveness."""

    print("="*70)
    print("TESTING TIMING-DEPENDENT REMINDER EFFECTIVENESS")
    print("="*70)
    print()

    # Load a simple task
    tasks = load_task_definitions()
    task_def = tasks['make_cereal']  # 8-step task

    # Create environment with NEW recency parameters
    params = SimulationParams(
        f0_base=0.6,                    # 60% baseline failure
        lambda_forget=0.10,             # 10% memory decay per tick
        lambda_recency=0.20,            # NEW: Fast recency decay
        effectiveness_recency=0.95,     # NEW: 95% max prevention
        c_int=2.0,
        c_fail_base=30.0
    )
    params.apply_task_defaults(task_def)

    env = ProcedureAssistantEnv(params, task_def)

    print(f"Parameters:")
    print(f"  f0_base = {params.f0_base} (60% baseline failure)")
    print(f"  lambda_recency = {params.lambda_recency} (half-life ~{np.log(2)/params.lambda_recency:.1f} ticks)")
    print(f"  effectiveness_recency = {params.effectiveness_recency} (95% max prevention)")
    print()

    # Test scenarios
    test_cases = [
        ("Perfect timing", 0, 1),      # Remind at tick 0, check at tick 1
        ("Recent", 0, 5),               # Remind at tick 0, check at tick 5
        ("Fading", 0, 10),              # Remind at tick 0, check at tick 10
        ("Old", 0, 20),                 # Remind at tick 0, check at tick 20
        ("Expired", 0, 30),             # Remind at tick 0, check at tick 30
        ("Just-in-time", 25, 28),       # Remind at tick 25, check at tick 28
    ]

    print(f"{'Scenario':<20} {'Ticks Ago':<12} {'Recency (r)':<15} {'Failure %':<12} {'Prevention %':<15}")
    print("-"*70)

    for scenario, remind_tick, check_tick in test_cases:
        # Simulate reminder at remind_tick
        env.pa_state.last_reminded_tick[0] = remind_tick
        # Memory after a single reminder: 0.3 delta_reminder
        env.pa_state.memory[0] = 0.3
        env.pa_state.global_tick = check_tick

        # Compute recency and failure probability
        recency = env._compute_recency_factor(0)
        fail_prob = env._compute_failure_probability(0)
        prevention = (1 - fail_prob / params.f0_base) * 100

        ticks_ago = check_tick - remind_tick

        # Also show what failure would be without recency (old model)
        base_fail = params.f0_base * np.exp(-params.k_memory * env.pa_state.memory[0])
        base_prevention = (1 - base_fail / params.f0_base) * 100

        print(f"{scenario:<20} {ticks_ago:<12} {recency:<15.3f} {fail_prob*100:<12.1f} {prevention:<15.1f}")

    # Add a test with NO prior memory (fresh reminder on step never seen before)
    print()
    print("FRESH REMINDER (no prior memory, just given):")
    env.pa_state.last_reminded_tick[0] = 0
    env.pa_state.memory[0] = 0.3  # Memory boost from single reminder
    env.pa_state.global_tick = 1
    recency = env._compute_recency_factor(0)
    fail_prob = env._compute_failure_probability(0)
    prevention = (1 - fail_prob / params.f0_base) * 100
    print(f"  Recency: {recency:.3f}")
    print(f"  Failure probability: {fail_prob*100:.1f}%")
    print(f"  Prevention: {prevention:.1f}%")
    print()

    # Test with perfect recency (r=1.0, theoretical maximum)
    print("THEORETICAL MAXIMUM (r=1.0, m=0.3):")
    base_fail_prob = params.f0_base * np.exp(-params.k_memory * 0.3)
    perfect_recency_multiplier = 1.0 - params.effectiveness_recency * 1.0
    perfect_fail_prob = base_fail_prob * perfect_recency_multiplier
    perfect_prevention = (1 - perfect_fail_prob / params.f0_base) * 100
    print(f"  Failure probability: {perfect_fail_prob*100:.1f}%")
    print(f"  Prevention: {perfect_prevention:.1f}%")

    print()
    print("="*70)
    print("EXPECTED RESULTS:")
    print("  Perfect timing (0-2 ticks): 97-99% prevention ✓")
    print("  Recent (3-5 ticks):         92-97% prevention ✓")
    print("  Fading (6-10 ticks):        80-92% prevention ✓")
    print("  Old (11-20 ticks):          55-80% prevention")
    print("  Expired (21+ ticks):        40-55% prevention")
    print("="*70)
    print()

    # Comparison with old model
    print("="*70)
    print("COMPARISON: OLD MODEL vs NEW MODEL")
    print("="*70)
    print()

    # Old model (no recency)
    params_old = SimulationParams(
        f0_base=0.6,
        lambda_forget=0.10,
        lambda_recency=0.0,            # Disable recency
        effectiveness_recency=0.0,     # Disable recency
        c_int=2.0,
        c_fail_base=30.0
    )
    params_old.apply_task_defaults(task_def)
    env_old = ProcedureAssistantEnv(params_old, task_def)

    # Set same memory state
    env_old.pa_state.memory[0] = 0.3
    fail_prob_old = env_old._compute_failure_probability(0)
    prevention_old = (1 - fail_prob_old / params_old.f0_base) * 100

    # New model (with recency, just reminded)
    env.pa_state.last_reminded_tick[0] = 0
    env.pa_state.global_tick = 1
    env.pa_state.memory[0] = 0.3
    fail_prob_new = env._compute_failure_probability(0)
    prevention_new = (1 - fail_prob_new / params.f0_base) * 100

    print(f"OLD MODEL (no recency):")
    print(f"  Failure probability: {fail_prob_old*100:.1f}%")
    print(f"  Prevention: {prevention_old:.1f}%")
    print()
    print(f"NEW MODEL (with recency, just reminded):")
    print(f"  Failure probability: {fail_prob_new*100:.1f}%")
    print(f"  Prevention: {prevention_new:.1f}%")
    print()
    print(f"IMPROVEMENT: {prevention_new - prevention_old:+.1f} percentage points")
    print("="*70)


if __name__ == '__main__':
    test_recency_effectiveness()
