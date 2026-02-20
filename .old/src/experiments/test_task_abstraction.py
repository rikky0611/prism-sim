"""
Unit tests for task abstraction system.

Verifies that the refactored simulation works with all 7 tasks.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

import numpy as np
from task_definitions import load_task_definitions, get_task_definition
from procedure_assistant_sim import (
    ProcedureAssistantEnv,
    SimulationParams,
    RandomAssistantPolicy,
    ProactiveReminderPolicy,
    ReactivePolicyHighCost,
    run_simulation,
)


def test_task_loading():
    """Test that all 7 tasks load correctly."""
    print("\n" + "="*80)
    print("TEST 1: Task Loading")
    print("="*80)

    tasks = load_task_definitions()
    assert len(tasks) == 7, f"Expected 7 tasks, got {len(tasks)}"

    expected_tasks = [
        "latte_making", "make_cereal", "make_coffee",
        "make_sandwich", "make_stencil", "make_tea", "cooking"
    ]

    for task_name in expected_tasks:
        assert task_name in tasks, f"Missing task: {task_name}"
        task_def = tasks[task_name]
        assert task_def.n_steps > 0, f"Task {task_name} has no steps"
        assert len(task_def.steps) == task_def.n_steps
        print(f"  ✓ {task_name}: {task_def.n_steps} steps, domain={task_def.domain}")

    print("\n✅ Task loading test PASSED")
    return True


def test_environment_initialization():
    """Test that environment initializes correctly for each task."""
    print("\n" + "="*80)
    print("TEST 2: Environment Initialization")
    print("="*80)

    tasks = load_task_definitions()
    params = SimulationParams()

    for task_name, task_def in tasks.items():
        # Apply task defaults
        params.apply_task_defaults(task_def)

        # Create environment
        env = ProcedureAssistantEnv(params, task_def)

        # Verify properties
        assert env.n_steps == task_def.n_steps
        assert len(env.procedural_steps) == task_def.n_steps
        assert len(env.assistant_actions) == 2 + task_def.n_steps

        # Verify action space
        assert 'silent' in env.assistant_actions
        assert 'confirm' in env.assistant_actions
        for i in range(task_def.n_steps):
            assert f'remind_{i}' in env.assistant_actions

        print(f"  ✓ {task_name}: n_steps={env.n_steps}, "
              f"n_actions={len(env.assistant_actions)}")

    print("\n✅ Environment initialization test PASSED")
    return True


def test_observation_action_space():
    """Test that observation and action spaces are correctly sized."""
    print("\n" + "="*80)
    print("TEST 3: Observation & Action Space Sizing")
    print("="*80)

    tasks = load_task_definitions()
    params = SimulationParams()

    for task_name, task_def in tasks.items():
        params.apply_task_defaults(task_def)
        env = ProcedureAssistantEnv(params, task_def)

        # Reset and get observation
        obs = env.reset()

        # Verify observation structure
        assert 'step_estimate' in obs
        assert 'elapsed_time' in obs
        assert 'step_name' in obs
        assert 'true_step' in obs
        assert 'memory' in obs

        # Verify memory vector size
        assert len(obs['memory']) == task_def.n_steps

        # Verify step estimates are in valid range
        assert 0 <= obs['step_estimate'] <= task_def.n_steps
        assert 0 <= obs['true_step'] <= task_def.n_steps

        print(f"  ✓ {task_name}: obs_memory_dim={len(obs['memory'])}, "
              f"action_space_size={len(env.assistant_actions)}")

    print("\n✅ Observation/action space test PASSED")
    return True


def test_episode_completion():
    """Test that episodes complete without errors."""
    print("\n" + "="*80)
    print("TEST 4: Episode Completion")
    print("="*80)

    tasks = load_task_definitions()
    params = SimulationParams()

    # Test with simplest task (make_cereal)
    task_def = tasks['make_cereal']
    params.apply_task_defaults(task_def)
    env = ProcedureAssistantEnv(params, task_def)

    # Create a random policy
    policy = RandomAssistantPolicy(task_def.n_steps)

    # Run a few steps
    obs = env.reset()
    max_steps = 500
    done = False

    for step in range(max_steps):
        action = policy.get_action(obs)
        obs, reward, done, info = env.step(action)

        if done:
            print(f"  ✓ Episode completed in {step+1} steps")
            print(f"    Total failures: {env.pa_state.total_failures}")
            print(f"    Total interactions: {env.pa_state.total_interactions}")
            print(f"    Total reward: {sum(env.history['rewards']):.2f}")
            break

    assert done, f"Episode did not complete within {max_steps} steps"

    print("\n✅ Episode completion test PASSED")
    return True


def test_policy_compatibility():
    """Test that policies work with different task sizes."""
    print("\n" + "="*80)
    print("TEST 5: Policy Compatibility")
    print("="*80)

    tasks = load_task_definitions()
    params = SimulationParams()

    # Test with smallest and largest tasks
    test_tasks = ['make_cereal', 'latte_making']

    for task_name in test_tasks:
        task_def = tasks[task_name]
        params.apply_task_defaults(task_def)

        print(f"\n  Testing with {task_name} ({task_def.n_steps} steps)...")

        # Test all three policy types
        policies = {
            'Random': RandomAssistantPolicy(task_def.n_steps),
            'Proactive': ProactiveReminderPolicy(task_def.n_steps),
            'Reactive': ReactivePolicyHighCost(task_def.n_steps, params=params),
        }

        for policy_name, policy in policies.items():
            env = ProcedureAssistantEnv(params, task_def)
            obs = env.reset()

            # Run 10 steps
            for _ in range(10):
                action = policy.get_action(obs)
                obs, reward, done, info = env.step(action)
                if done:
                    break

            print(f"    ✓ {policy_name} policy works")

    print("\n✅ Policy compatibility test PASSED")
    return True


def test_cross_task_comparison():
    """Test run_simulation with multiple tasks."""
    print("\n" + "="*80)
    print("TEST 6: Cross-Task Simulation")
    print("="*80)

    tasks = load_task_definitions()
    params = SimulationParams()

    # Test with 3 tasks of different sizes
    test_tasks = ['make_cereal', 'make_tea', 'cooking']

    for task_name in test_tasks:
        task_def = tasks[task_name]
        params.apply_task_defaults(task_def)

        # Create random policy
        policy = RandomAssistantPolicy(task_def.n_steps)

        # Run simulation
        results = run_simulation(
            policy=policy,
            params=params,
            task_def=task_def,
            n_episodes=5,
            verbose=False
        )

        # Verify results
        assert len(results['total_rewards']) == 5
        assert len(results['total_failures']) == 5
        assert len(results['total_interactions']) == 5

        mean_reward = np.mean(results['total_rewards'])
        mean_failures = np.mean(results['total_failures'])

        print(f"  ✓ {task_name}: mean_reward={mean_reward:.2f}, "
              f"mean_failures={mean_failures:.2f}")

    print("\n✅ Cross-task simulation test PASSED")
    return True


def test_criticality_values():
    """Test that per-step criticality affects failure costs."""
    print("\n" + "="*80)
    print("TEST 7: Step Criticality")
    print("="*80)

    # Test with make_stencil (has high criticality steps)
    task_def = get_task_definition("make_stencil")

    print(f"\n  Testing {task_def.task_name}:")
    print(f"  Base failure cost: {task_def.base_failure_cost}")

    # Find steps with varying criticality
    for i, step in enumerate(task_def.steps):
        if step.criticality != 1.0:
            fail_cost = task_def.get_step_failure_cost(i)
            print(f"    Step {i} ({step.name}): criticality={step.criticality:.1f}, "
                  f"fail_cost={fail_cost:.1f}")

    # Verify critical steps have higher costs
    critical_step_idx = 2  # check_exhaust (criticality=2.5)
    normal_step_idx = 0    # design_stencil (criticality=1.0)

    critical_cost = task_def.get_step_failure_cost(critical_step_idx)
    normal_cost = task_def.get_step_failure_cost(normal_step_idx)

    assert critical_cost > normal_cost, "Critical steps should have higher failure costs"
    print(f"\n  ✓ Critical step cost ({critical_cost:.1f}) > Normal step cost ({normal_cost:.1f})")

    print("\n✅ Criticality test PASSED")
    return True


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "#"*80)
    print("# TASK ABSTRACTION UNIT TESTS")
    print("#"*80)

    tests = [
        test_task_loading,
        test_environment_initialization,
        test_observation_action_space,
        test_episode_completion,
        test_policy_compatibility,
        test_cross_task_comparison,
        test_criticality_values,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n❌ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "#"*80)
    print(f"# TEST SUMMARY: {passed}/{len(tests)} passed, {failed} failed")
    print("#"*80)

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! System is ready for training.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Fix issues before proceeding.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
