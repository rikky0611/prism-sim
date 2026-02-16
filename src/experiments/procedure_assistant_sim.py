"""
Procedure Assistant Simulation using Overcooked AI

This implements the procedure assistant POMDP formulation from modeling.pdf
using Overcooked AI as the underlying task environment.

Key design decisions:
1. Human agent operates in the kitchen (standard Overcooked agent)
2. Assistant observes from outside (partial observability via noisy sensors)
3. Procedural steps extracted from recipe completion (onion soup)
4. Memory mechanism affects failure probability at each step
5. Interaction costs model interruption burden
"""

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json

# Import task definitions for multi-task support
try:
    from task_definitions import TaskDefinition
except ImportError:
    # Fallback for when running from different directory
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from task_definitions import TaskDefinition


# ============================================================================
# NOTE: PROCEDURAL STEPS ARE NOW DEFINED PER-TASK
# ============================================================================
# The hardcoded PROCEDURAL_STEPS, N_STEPS, and ASSISTANT_ACTIONS have been
# removed. These are now dynamically generated based on TaskDefinition.


# ============================================================================
# PROCEDURE ASSISTANT STATE
# ============================================================================
class ProcedureAssistantState:
    """
    Latent state: x_t = (s_t, tau_t, m_t)
    - s_t: current procedural step (or 'done')
    - tau_t: elapsed time in current step
    - m_t: reminder memory for each step (N-dimensional vector)
    """
    def __init__(self, n_steps: int):
        self.current_step: int = 0  # index into PROCEDURAL_STEPS
        self.tau: int = 0  # elapsed time in step
        self.memory: np.ndarray = np.zeros(n_steps)  # memory for each step
        self.is_done: bool = False
        self.total_failures: int = 0
        self.total_interactions: int = 0

    def copy(self):
        new_state = ProcedureAssistantState(len(self.memory))
        new_state.current_step = self.current_step
        new_state.tau = self.tau
        new_state.memory = self.memory.copy()
        new_state.is_done = self.is_done
        new_state.total_failures = self.total_failures
        new_state.total_interactions = self.total_interactions
        return new_state


# ============================================================================
# ASSISTANT ACTIONS
# ============================================================================
# NOTE: ASSISTANT_ACTIONS is now dynamically built per-task in the environment
# Human actions remain constant across all tasks
HUMAN_ACTIONS = {
    'silent': 0,
    'narrate': 1,
    'respond': 2,
}


# ============================================================================
# PARAMETERS FOR SIMULATION
# ============================================================================
class SimulationParams:
    """Parameters for the procedure assistant simulation"""
    def __init__(
        self,
        # Memory dynamics
        lambda_forget: float = 0.05,      # Forgetting rate per tick
        delta_reminder: float = 0.3,       # Memory boost from reminder

        # Failure model
        f0_base: float = 0.3,             # Base failure probability
        k_memory: float = 2.0,            # Memory effect on failure

        # Cost structure
        c_int: float = 5.0,               # Interruption cost
        c_nar: float = 1.0,               # Narration cost (human-initiated)
        c_resp: float = 2.0,              # Response cost (to confirm)
        c_fail_base: float = 20.0,        # Base failure cost

        # Step-specific failure costs (severity)
        step_failure_costs: Optional[Dict[str, float]] = None,

        # Human bounded rationality
        beta: float = 1.0,                # Rationality parameter

        # Observation noise
        obs_noise: float = 0.2,           # Probability of wrong step observation

        # Step duration parameters
        step_mean_duration: int = 30,     # Mean ticks per step
        step_std_duration: int = 10,      # Std dev of step duration
    ):
        self.lambda_forget = lambda_forget
        self.delta_reminder = delta_reminder
        self.f0_base = f0_base
        self.k_memory = k_memory
        self.c_int = c_int
        self.c_nar = c_nar
        self.c_resp = c_resp
        self.c_fail_base = c_fail_base
        self.beta = beta
        self.obs_noise = obs_noise
        self.step_mean_duration = step_mean_duration
        self.step_std_duration = step_std_duration

        # Step-specific failure costs (will be set by task definition)
        # This is kept for backward compatibility but will be overridden
        # by TaskDefinition.get_step_failure_cost()
        self.step_failure_costs = step_failure_costs or {}

    def apply_task_defaults(self, task_def: TaskDefinition):
        """Apply task-specific default costs from TaskDefinition.

        Args:
            task_def: TaskDefinition with base costs and step criticalities
        """
        self.c_fail_base = task_def.base_failure_cost
        self.c_int = task_def.interruption_cost

    def to_dict(self):
        """Convert parameters to dictionary for logging"""
        return {
            'lambda_forget': self.lambda_forget,
            'delta_reminder': self.delta_reminder,
            'f0_base': self.f0_base,
            'k_memory': self.k_memory,
            'c_int': self.c_int,
            'c_nar': self.c_nar,
            'c_resp': self.c_resp,
            'c_fail_base': self.c_fail_base,
            'beta': self.beta,
            'obs_noise': self.obs_noise,
            'step_mean_duration': self.step_mean_duration,
            'step_std_duration': self.step_std_duration,
        }


# ============================================================================
# PROCEDURE ASSISTANT ENVIRONMENT
# ============================================================================
class ProcedureAssistantEnv:
    """
    Procedural task environment implementing the procedure assistant dynamics.

    Supports multiple procedural tasks through TaskDefinition abstraction.

    Key features:
    - Variable-length tasks (8-20 steps)
    - Per-step criticality values affecting failure costs
    - Dynamic observation and action spaces based on task size
    - Memory dynamics and failure model unchanged from original

    Args:
        params: SimulationParams with memory/failure/cost parameters
        task_def: TaskDefinition specifying the procedural task
    """
    def __init__(self, params: SimulationParams, task_def: TaskDefinition):
        self.params = params
        self.task_def = task_def

        # Task-specific properties
        self.n_steps = task_def.n_steps
        self.procedural_steps = task_def.step_names

        # Build action space dynamically based on task size
        self.assistant_actions = self._build_action_space()

        # Procedure assistant state
        self.pa_state = ProcedureAssistantState(self.n_steps)

        # Step duration tracking (using discrete hazard model)
        self.step_target_duration = {}
        self._sample_step_durations()

    def _build_action_space(self) -> Dict[str, int]:
        """Build assistant action space dynamically based on task size.

        Returns:
            Dictionary mapping action names to action IDs.
            Format: {'silent': 0, 'confirm': 1, 'remind_0': 2, ..., 'remind_N-1': N+1}
        """
        actions = {
            'silent': 0,
            'confirm': 1,
        }
        for i in range(self.n_steps):
            actions[f'remind_{i}'] = 2 + i
        return actions

        # History tracking
        self.history = {
            'states': [],
            'observations': [],
            'actions_assistant': [],
            'actions_human': [],
            'rewards': [],
            'failures': [],
            'step_progression': [],
        }

    def _sample_step_durations(self):
        """Sample target durations for each step from truncated normal.

        Uses per-step duration parameters from TaskDefinition if available,
        otherwise falls back to global parameters.
        """
        for step_def in self.task_def.steps:
            duration = max(5, int(np.random.normal(
                step_def.mean_duration,
                step_def.std_duration
            )))
            self.step_target_duration[step_def.name] = duration

    def reset(self):
        """Reset environment and procedure state"""
        self.pa_state = ProcedureAssistantState(self.n_steps)
        self._sample_step_durations()

        self.history = {
            'states': [],
            'observations': [],
            'actions_assistant': [],
            'actions_human': [],
            'rewards': [],
            'failures': [],
            'step_progression': [],
        }

        return self._get_observation()

    def _get_observation(self) -> Dict:
        """
        Assistant's observation: (o_t, z_t)
        - o_t: Noisy sensor observation of current step and elapsed time
        - z_t: Linguistic signal (if human communicates)
        """
        true_step = self.pa_state.current_step

        # Handle done state
        if self.pa_state.is_done or true_step >= self.n_steps:
            obs = {
                'step_estimate': self.n_steps,
                'elapsed_time': self.pa_state.tau,
                'step_name': 'done',
                'true_step': true_step,
                'memory': self.pa_state.memory.copy(),
            }
            return obs

        # Add observation noise (confuse current step)
        if np.random.random() < self.params.obs_noise:
            observed_step = np.random.randint(self.n_steps)
        else:
            observed_step = true_step

        obs = {
            'step_estimate': observed_step,
            'elapsed_time': self.pa_state.tau,
            'step_name': self.procedural_steps[observed_step],
            'true_step': true_step,  # Ground truth (for evaluation)
            'memory': self.pa_state.memory.copy(),
        }

        return obs

    def _compute_failure_probability(self, step_idx: int) -> float:
        """
        Failure probability decreases with memory:
        f_n(m) = f0_base * exp(-k * m)
        """
        memory = self.pa_state.memory[step_idx]
        prob = self.params.f0_base * np.exp(-self.params.k_memory * memory)
        return np.clip(prob, 0.0, 1.0)

    def _check_step_completion(self) -> bool:
        """
        Discrete hazard model: probability of completing step at time tau
        h_s(tau) = Pr(D_s = tau+1 | D_s > tau)

        Simplified: use exponential-like hazard that increases with time
        """
        current_step_name = self.procedural_steps[self.pa_state.current_step]
        target_duration = self.step_target_duration[current_step_name]

        # Hazard increases as we approach target duration
        # After target, high probability of completion
        if self.pa_state.tau >= target_duration:
            hazard = 0.8  # High probability once we reach expected duration
        else:
            # Gradually increasing hazard
            hazard = 0.05 + 0.3 * (self.pa_state.tau / target_duration)

        return np.random.random() < hazard

    def _update_memory(self, assistant_action: int):
        """
        Update memory according to Eq. 3:
        m_{n,t+1} = (1 - lambda) * m_{n,t} + Delta_A * I[a_t = remind_n]
        """
        # Decay all memories
        self.pa_state.memory *= (1 - self.params.lambda_forget)

        # If assistant gave a reminder, boost that step's memory
        for step_idx in range(self.n_steps):
            remind_action_id = self.assistant_actions[f'remind_{step_idx}']
            if assistant_action == remind_action_id:
                self.pa_state.memory[step_idx] += self.params.delta_reminder

        # Clip memories to reasonable range
        self.pa_state.memory = np.clip(self.pa_state.memory, 0, 2.0)

    def _sample_human_action(self, assistant_action: int, expected_value: float) -> int:
        """
        Human decides whether to communicate based on utility (Eqs. 5-6)
        P(narrate | silent) = sigma(beta * (g_t - c_nar))
        P(respond | confirm) = sigma(beta * (g_t - c_resp - c_int))
        """
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Determine feasible actions
        if assistant_action == self.assistant_actions['silent']:
            # Can narrate or stay silent
            utility = expected_value - self.params.c_nar
            prob_communicate = sigmoid(self.params.beta * utility)
            return HUMAN_ACTIONS['narrate'] if np.random.random() < prob_communicate else HUMAN_ACTIONS['silent']

        elif assistant_action == self.assistant_actions['confirm']:
            # Can respond or stay silent
            utility = expected_value - self.params.c_resp - self.params.c_int
            prob_communicate = sigmoid(self.params.beta * utility)
            return HUMAN_ACTIONS['respond'] if np.random.random() < prob_communicate else HUMAN_ACTIONS['silent']

        else:
            # Reminder given, human stays silent
            return HUMAN_ACTIONS['silent']

    def _compute_expected_value_of_communication(self) -> float:
        """
        Estimate g_t: expected reduction in failure cost from revealing current step.
        Simplified: return current failure probability * failure cost
        """
        current_step = self.pa_state.current_step
        fail_prob = self._compute_failure_probability(current_step)
        fail_cost = self.task_def.get_step_failure_cost(current_step)
        return fail_prob * fail_cost

    def step(self, assistant_action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one tick of the environment.

        Args:
            assistant_action: Action index from ASSISTANT_ACTIONS

        Returns:
            observation, reward, done, info
        """
        # Sample human interaction action based on utility
        expected_value = self._compute_expected_value_of_communication()
        human_action = self._sample_human_action(assistant_action, expected_value)

        # Update memory based on assistant action
        self._update_memory(assistant_action)

        # Check if current step completes
        step_completed = self._check_step_completion()

        failure_occurred = False
        reward = 0.0

        if step_completed:
            # Sample failure based on memory-dependent probability
            fail_prob = self._compute_failure_probability(self.pa_state.current_step)
            failure_occurred = np.random.random() < fail_prob

            if failure_occurred:
                fail_cost = self.task_def.get_step_failure_cost(self.pa_state.current_step)
                reward -= fail_cost
                self.pa_state.total_failures += 1

            # Progress to next step
            self.pa_state.current_step += 1
            self.pa_state.tau = 0

            if self.pa_state.current_step >= self.n_steps:
                self.pa_state.is_done = True
        else:
            # Stay in same step, increment time
            self.pa_state.tau += 1

        # Add interaction costs (Eq. 7)
        if assistant_action != self.assistant_actions['silent']:
            reward -= self.params.c_int
            self.pa_state.total_interactions += 1

        if human_action == HUMAN_ACTIONS['narrate']:
            reward -= self.params.c_nar

        if human_action == HUMAN_ACTIONS['respond']:
            reward -= self.params.c_resp

        # Record history
        self.history['states'].append(self.pa_state.copy())
        self.history['actions_assistant'].append(assistant_action)
        self.history['actions_human'].append(human_action)
        self.history['rewards'].append(reward)
        self.history['failures'].append(failure_occurred)
        self.history['step_progression'].append(
            (self.pa_state.current_step, self.pa_state.tau)
        )

        obs = self._get_observation()
        self.history['observations'].append(obs)

        return obs, reward, self.pa_state.is_done, {
            'failure': failure_occurred,
            'step_completed': step_completed,
            'human_action': human_action,
        }


# ============================================================================
# SIMPLE POLICIES FOR TESTING
# ============================================================================
class RandomAssistantPolicy:
    """Random baseline: randomly choose actions.

    Args:
        n_steps: Number of steps in the task
        action_probs: Optional dictionary of action probabilities
    """
    def __init__(self, n_steps: int, action_probs: Optional[Dict[str, float]] = None):
        self.n_steps = n_steps
        self.n_actions = 2 + n_steps  # silent, confirm, remind_0, ..., remind_N-1

        if action_probs is None:
            # Default: 70% silent, 10% confirm, remaining split across reminders
            reminder_prob = 0.20 / n_steps if n_steps > 0 else 0.0
            self.action_probs = np.array([0.7] + [reminder_prob] * n_steps + [0.1])
            self.action_probs = self.action_probs[:self.n_actions]
            self.action_probs /= self.action_probs.sum()
        else:
            # Build action space for this task size
            assistant_actions = {'silent': 0, 'confirm': 1}
            for i in range(n_steps):
                assistant_actions[f'remind_{i}'] = 2 + i

            probs = []
            for action_name in sorted(assistant_actions.keys(), key=assistant_actions.get):
                probs.append(action_probs.get(action_name, 0.0))
            self.action_probs = np.array(probs)
            self.action_probs /= self.action_probs.sum()

    def get_action(self, obs):
        return np.random.choice(self.n_actions, p=self.action_probs)


class ProactiveReminderPolicy:
    """Proactive policy: Send reminders based on memory threshold.

    If memory for upcoming step is low, send reminder.

    Args:
        n_steps: Number of steps in the task
        memory_threshold: Memory level below which to send reminder
        lookahead: Number of steps ahead to check
    """
    def __init__(self, n_steps: int, memory_threshold: float = 0.3, lookahead: int = 1):
        self.n_steps = n_steps
        self.memory_threshold = memory_threshold
        self.lookahead = lookahead

        # Build action space for this task size
        self.assistant_actions = {'silent': 0, 'confirm': 1}
        for i in range(n_steps):
            self.assistant_actions[f'remind_{i}'] = 2 + i

    def get_action(self, obs):
        current_step = obs['step_estimate']
        memory = obs['memory']

        # If done, stay silent
        if current_step >= self.n_steps or obs['step_name'] == 'done':
            return self.assistant_actions['silent']

        # Check if we should remind about upcoming steps
        for offset in range(self.lookahead):
            next_step = current_step + offset
            if next_step < self.n_steps and memory[next_step] < self.memory_threshold:
                return self.assistant_actions[f'remind_{next_step}']

        # Otherwise stay silent
        return self.assistant_actions['silent']


class ReactivePolicyHighCost:
    """Reactive policy for high interruption cost.

    Only remind when failure risk is very high.

    Args:
        n_steps: Number of steps in the task
        risk_threshold: Failure probability above which to intervene
        params: SimulationParams for failure model
    """
    def __init__(self, n_steps: int, risk_threshold: float = 0.25, params: SimulationParams = None):
        self.n_steps = n_steps
        self.risk_threshold = risk_threshold
        self.params = params or SimulationParams()

        # Build action space for this task size
        self.assistant_actions = {'silent': 0, 'confirm': 1}
        for i in range(n_steps):
            self.assistant_actions[f'remind_{i}'] = 2 + i

    def get_action(self, obs):
        current_step = obs['step_estimate']
        memory = obs['memory']

        # If done, stay silent
        if current_step >= self.n_steps or obs['step_name'] == 'done':
            return self.assistant_actions['silent']

        # Compute current failure probability
        fail_prob = self.params.f0_base * np.exp(-self.params.k_memory * memory[current_step])

        # Only intervene if risk is very high
        if fail_prob > self.risk_threshold:
            return self.assistant_actions[f'remind_{current_step}']

        return self.assistant_actions['silent']


# ============================================================================
# SIMULATION RUNNER
# ============================================================================
def run_simulation(
    policy,
    params: SimulationParams,
    task_def: TaskDefinition,
    n_episodes: int = 10,
    verbose: bool = True
) -> Dict:
    """Run multiple episodes and collect statistics.

    Args:
        policy: Policy object with get_action(obs) method
        params: SimulationParams with memory/failure/cost parameters
        task_def: TaskDefinition specifying the procedural task
        n_episodes: Number of episodes to run
        verbose: Whether to print progress

    Returns:
        Dictionary with episode statistics
    """
    env = ProcedureAssistantEnv(params, task_def)

    results = {
        'total_rewards': [],
        'total_failures': [],
        'total_interactions': [],
        'episode_lengths': [],
        'histories': [],
    }

    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

        results['total_rewards'].append(episode_reward)
        results['total_failures'].append(env.pa_state.total_failures)
        results['total_interactions'].append(env.pa_state.total_interactions)
        results['episode_lengths'].append(len(env.history['rewards']))
        results['histories'].append(env.history)

        if verbose:
            print(f"Episode {episode+1}/{n_episodes}: "
                  f"Reward={episode_reward:.1f}, "
                  f"Failures={env.pa_state.total_failures}, "
                  f"Interactions={env.pa_state.total_interactions}")

    # Compute summary statistics
    results['mean_reward'] = np.mean(results['total_rewards'])
    results['std_reward'] = np.std(results['total_rewards'])
    results['mean_failures'] = np.mean(results['total_failures'])
    results['mean_interactions'] = np.mean(results['total_interactions'])

    return results


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_results(results_dict: Dict[str, Dict], save_path: str = None):
    """Plot comparison of different policies/parameters"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    labels = list(results_dict.keys())

    # Mean rewards
    ax = axes[0, 0]
    means = [results_dict[label]['mean_reward'] for label in labels]
    stds = [results_dict[label]['std_reward'] for label in labels]
    ax.bar(labels, means, yerr=stds, capsize=5)
    ax.set_ylabel('Mean Total Reward')
    ax.set_title('Total Reward (higher is better)')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Mean failures
    ax = axes[0, 1]
    failures = [results_dict[label]['mean_failures'] for label in labels]
    ax.bar(labels, failures)
    ax.set_ylabel('Mean Failures per Episode')
    ax.set_title('Failure Count (lower is better)')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Mean interactions
    ax = axes[1, 0]
    interactions = [results_dict[label]['mean_interactions'] for label in labels]
    ax.bar(labels, interactions)
    ax.set_ylabel('Mean Interactions per Episode')
    ax.set_title('Interaction Count')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Reward distribution
    ax = axes[1, 1]
    data = [results_dict[label]['total_rewards'] for label in labels]
    ax.boxplot(data, labels=labels)
    ax.set_ylabel('Total Reward')
    ax.set_title('Reward Distribution')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.savefig('simulation_results.png', dpi=150, bbox_inches='tight')
        print("Plot saved to simulation_results.png")

    return fig


def plot_trajectory(history: Dict, save_path: str = None):
    """Plot a single episode trajectory"""

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    timesteps = range(len(history['rewards']))

    # Step progression
    ax = axes[0]
    steps = [s for s, _ in history['step_progression']]
    ax.plot(timesteps, steps, marker='o', markersize=2)
    ax.set_ylabel('Procedural Step')
    ax.set_title('Step Progression Over Time')
    ax.set_yticks(range(N_STEPS))
    ax.set_yticklabels(PROCEDURAL_STEPS, fontsize=8)
    ax.grid(alpha=0.3)

    # Actions and failures
    ax = axes[1]
    assistant_actions = history['actions_assistant']
    failures = np.array(history['failures'], dtype=float)

    # Mark interactions (non-silent actions)
    interactions = [1 if a != ASSISTANT_ACTIONS['silent'] else 0 for a in assistant_actions]
    ax.fill_between(timesteps, 0, interactions, alpha=0.3, label='Assistant Interaction', color='blue')

    # Mark failures
    failure_times = [t for t, f in enumerate(failures) if f]
    ax.scatter(failure_times, [0.5] * len(failure_times), color='red', marker='x', s=100, label='Failure', zorder=5)

    ax.set_ylabel('Events')
    ax.set_ylim(-0.1, 1.1)
    ax.legend()
    ax.set_title('Interactions and Failures')
    ax.grid(alpha=0.3)

    # Cumulative reward
    ax = axes[2]
    cumulative_reward = np.cumsum(history['rewards'])
    ax.plot(timesteps, cumulative_reward, color='green')
    ax.set_xlabel('Time (ticks)')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Cumulative Reward Over Time')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Trajectory plot saved to {save_path}")
    else:
        plt.savefig('trajectory.png', dpi=150, bbox_inches='tight')
        print("Trajectory plot saved to trajectory.png")

    return fig


if __name__ == "__main__":
    print("Procedure Assistant Simulation")
    print("=" * 60)
    print()

    # Test the environment
    print("Testing environment...")
    params = SimulationParams()
    env = ProcedureAssistantEnv(params)
    obs = env.reset()
    print(f"Initial observation: step={obs['step_name']}, tau={obs['elapsed_time']}")
    print(f"Memory state: {obs['memory']}")
    print()
