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
    Latent state: x_t = (s_t, tau_t, m_t, r_t)
    - s_t: current procedural step (or 'done')
    - tau_t: elapsed time in current step
    - m_t: reminder memory for each step (N-dimensional vector, long-term)
    - r_t: last reminded tick for each step (for recency calculation)
    """
    def __init__(self, n_steps: int):
        self.current_step: int = 0  # index into PROCEDURAL_STEPS
        self.tau: int = 0  # elapsed time in step
        self.memory: np.ndarray = np.zeros(n_steps)  # base memory for each step
        self.last_reminded_tick: np.ndarray = np.full(n_steps, -999)  # NEW: Track reminder timing
        self.global_tick: int = 0  # NEW: Global time counter
        self.is_done: bool = False
        self.total_failures: int = 0
        self.total_interactions: int = 0
        self.total_responses: int = 0  # NEW: Track human responses to interruptions
        self.total_narrations: int = 0  # NEW: Track human narrations

    def copy(self):
        new_state = ProcedureAssistantState(len(self.memory))
        new_state.current_step = self.current_step
        new_state.tau = self.tau
        new_state.memory = self.memory.copy()
        new_state.last_reminded_tick = self.last_reminded_tick.copy()  # NEW
        new_state.global_tick = self.global_tick  # NEW
        new_state.is_done = self.is_done
        new_state.total_failures = self.total_failures
        new_state.total_interactions = self.total_interactions
        new_state.total_responses = self.total_responses  # NEW
        new_state.total_narrations = self.total_narrations  # NEW
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
        lambda_forget: float = 0.03,      # Forgetting rate per tick (23-tick half-life → slower decay)
        delta_reminder: float = 0.8,       # Memory boost from reminder (stronger, longer-lasting protection)

        # Failure model
        f0_base: float = 0.6,             # Base failure probability (60% baseline when m=0) - INCREASED for better learning
        k_memory: float = 3.0,            # Memory effect on failure (steeper curve, emphasizes timing)

        # Cost structure (UPDATED: per-step costs, fixed c_int)
        c_fail_per_step: Optional[np.ndarray] = None,  # NEW: Per-step failure costs array
        c_int: float = 1.0,               # Interruption cost (FIXED to 1.0)
        c_nar: float = 1.0,               # Narration cost (human-initiated)
        c_resp: float = 2.0,              # Response cost (to confirm)
        c_off_timing: float = 0.5,        # Penalty for reminding wrong step (off-timing)

        # Deprecated parameters (backward compatibility)
        c_fail_base: Optional[float] = None,  # DEPRECATED: Use c_fail_per_step instead
        step_failure_costs: Optional[Dict[str, float]] = None,  # DEPRECATED

        # Human bounded rationality
        beta: float = 1.0,                # Rationality parameter

        # Observation noise
        obs_noise: float = 0.2,           # Probability of wrong step observation

        # Step duration parameters
        step_mean_duration: int = 30,     # Mean ticks per step
        step_std_duration: int = 10,      # Std dev of step duration

        # NEW: Recency-based reminder effectiveness
        lambda_recency: float = 0.20,     # Fast recency decay (half-life ~3.5 ticks)
        effectiveness_recency: float = 0.95,  # Max additional prevention from recent reminder (95%)
    ):
        self.lambda_forget = lambda_forget
        self.delta_reminder = delta_reminder
        self.f0_base = f0_base
        self.k_memory = k_memory
        self.c_int = c_int  # Fixed to 1.0
        self.c_nar = c_nar
        self.c_resp = c_resp
        self.c_off_timing = c_off_timing  # Penalty for off-timing reminders
        self.beta = beta
        self.obs_noise = obs_noise
        self.step_mean_duration = step_mean_duration
        self.step_std_duration = step_std_duration
        self.lambda_recency = lambda_recency
        self.effectiveness_recency = effectiveness_recency

        # Per-step failure costs (NEW primary mechanism)
        if c_fail_per_step is not None:
            self.c_fail_per_step = c_fail_per_step
        elif c_fail_base is not None:
            # Backward compatibility: uniform costs for all steps
            print(f"WARNING: c_fail_base is deprecated. Using uniform cost {c_fail_base} for all steps.")
            self.c_fail_per_step = np.full(8, c_fail_base)  # Default 8 steps
        else:
            # Default: uniform costs
            self.c_fail_per_step = np.full(8, 20.0)

        # Keep deprecated fields for backward compatibility
        self.c_fail_base = c_fail_base  # DEPRECATED
        self.step_failure_costs = step_failure_costs or {}  # DEPRECATED

    # REMOVED: apply_task_defaults() - no longer needed with per-step costs

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
            'c_fail_per_step': self.c_fail_per_step.tolist() if isinstance(self.c_fail_per_step, np.ndarray) else self.c_fail_per_step,
            'beta': self.beta,
            'obs_noise': self.obs_noise,
            'step_mean_duration': self.step_mean_duration,
            'step_std_duration': self.step_std_duration,
            'lambda_recency': self.lambda_recency,
            'effectiveness_recency': self.effectiveness_recency,
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
        """Build assistant action space dynamically based on CRITICAL steps only.

        Only steps with criticality > 0 get reminder actions.
        This creates a sparse action space focusing on failure-prone steps.

        Returns:
            Dictionary mapping action names to action IDs.
            Format: {'silent': 0, 'confirm': 1, 'remind_3': 2, 'remind_4': 3}
            (example for make_cereal with only steps 3-4 critical)
        """
        actions = {
            'silent': 0,
            'confirm': 1,
        }

        action_id = 2
        for i, step in enumerate(self.task_def.steps):
            if step.criticality > 0:  # Only critical steps get reminders
                actions[f'remind_{i}'] = action_id
                action_id += 1

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

        # Add Gaussian observation noise centered on true step
        if np.random.random() < self.params.obs_noise:
            # Gaussian noise with std=1.0, centered on true step
            noise = np.random.normal(0, 1.0)
            observed_step = int(np.clip(true_step + noise, 0, self.n_steps - 1))
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
        Failure probability with base memory AND recency bonus.

        f(m, r) = f0_base × exp(-k × m) × (1 - ε_recency × r)

        where:
        - m: base memory (long-term procedural knowledge, decays slowly)
        - r: recency factor (short-term reminder freshness, decays fast)
        - ε_recency: maximum additional prevention from recent reminder (default 0.95)

        Examples:
        - Just reminded (r=1.0, m=0.3): f = 0.6 × 0.55 × 0.05 = 0.016 (1.6% failure, 98% prevention)
        - 10 ticks ago (r=0.5, m=0.2): f = 0.6 × 0.67 × 0.5 = 0.20 (20% failure, 80% prevention)
        - No recent reminder (r=0, m=0): f = 0.6 × 1.0 × 1.0 = 0.60 (60% failure, baseline)
        """
        # Non-critical steps (criticality=0) cannot fail - you can't fail at trivial actions
        if step_idx < len(self.task_def.steps):
            if self.task_def.steps[step_idx].criticality == 0:
                return 0.0

        # Base failure probability (exponential decay with long-term memory)
        memory = self.pa_state.memory[step_idx]
        base_failure_prob = self.params.f0_base * np.exp(-self.params.k_memory * memory)

        # Recency bonus (strong for recent reminders, decays rapidly)
        recency_factor = self._compute_recency_factor(step_idx)
        recency_multiplier = 1.0 - self.params.effectiveness_recency * recency_factor

        # Combined failure probability
        final_prob = base_failure_prob * recency_multiplier
        return np.clip(final_prob, 0.0, 1.0)

    def _compute_recency_factor(self, step_idx: int) -> float:
        """Compute recency factor: 1.0 if just reminded, decays exponentially.

        Args:
            step_idx: Index of the step

        Returns:
            Recency factor in [0, 1], where 1.0 = just reminded, 0.0 = no recent reminder
        """
        ticks_since_reminder = self.pa_state.global_tick - self.pa_state.last_reminded_tick[step_idx]

        # If never reminded or very old, return 0
        if ticks_since_reminder > 50:
            return 0.0

        # Exponential decay: r = exp(-λ × t)
        recency = np.exp(-self.params.lambda_recency * ticks_since_reminder)
        return np.clip(recency, 0.0, 1.0)

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
        Update memory and track reminder timing.

        Base memory update: m_{n,t+1} = (1 - lambda) * m_{n,t} + Delta_A * I[a_t = remind_n]
        Timing update: record global_tick when reminder given for recency calculation
        """
        # Decay all base memories
        self.pa_state.memory *= (1 - self.params.lambda_forget)

        # If assistant gave a reminder, boost memory AND record timing
        for step_idx in range(self.n_steps):
            remind_key = f'remind_{step_idx}'
            if remind_key in self.assistant_actions:  # Only check if this step has a reminder action
                remind_action_id = self.assistant_actions[remind_key]
                if assistant_action == remind_action_id:
                    self.pa_state.memory[step_idx] += self.params.delta_reminder
                    self.pa_state.last_reminded_tick[step_idx] = self.pa_state.global_tick  # NEW

        # Clip memories to reasonable range
        self.pa_state.memory = np.clip(self.pa_state.memory, 0, 2.0)

        # Increment global time counter
        self.pa_state.global_tick += 1  # NEW

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
                # Use per-step failure cost from params
                step_idx = self.pa_state.current_step
                fail_cost = self.params.c_fail_per_step[step_idx]
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

            # OFF-TIMING PENALTY: If this is a reminder for wrong step, add extra penalty
            if assistant_action != self.assistant_actions['confirm']:
                # This is a remind_N action - find which step N
                reminded_step = None
                for step_name, action_id in self.assistant_actions.items():
                    if action_id == assistant_action and step_name.startswith('remind_'):
                        reminded_step = int(step_name.split('_')[1])
                        break

                if reminded_step is not None:
                    current_step = self.pa_state.current_step
                    # Penalize if reminder doesn't match current or immediate next step
                    if reminded_step not in [current_step, current_step + 1]:
                        reward -= self.params.c_off_timing

        if human_action == HUMAN_ACTIONS['narrate']:
            reward -= self.params.c_nar
            self.pa_state.total_narrations += 1  # Track narration count

        if human_action == HUMAN_ACTIONS['respond']:
            reward -= self.params.c_resp
            self.pa_state.total_responses += 1  # Track response count

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
