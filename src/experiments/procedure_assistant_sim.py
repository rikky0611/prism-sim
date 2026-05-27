"""
Single-agent Procedure Assistant POMDP simulation.

Models an AI assistant that observes a human performing a procedural task
and must decide when to intervene (remind/confirm) vs stay silent,
balancing interruption costs against failure costs.

Key design decisions:
1. Partial observability: assistant sees noisy step and memory estimates
2. Per-step failure costs derived from task step criticality
3. Memory mechanism affects failure probability at each step
4. Interaction costs model interruption burden
5. Action space: silent, confirm, remind_i (critical steps only)
"""

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json

# Import task definitions for multi-task support
try:
    from task_definitions import TaskDefinition, get_task_definition
except ImportError:
    # Fallback for when running from different directory
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from task_definitions import TaskDefinition, get_task_definition


# ============================================================================
# PROCEDURE ASSISTANT STATE
# ============================================================================
class ProcedureAssistantState:
    """
    Latent state: x_t = (s_t, tau_t, m_t)
    - s_t: current procedural step (or 'done')
    - tau_t: elapsed time in current step
    - m_t: reminder memory for each step (N-dimensional vector, long-term)
    """
    def __init__(self, n_steps: int):
        self.current_step: int = 0
        self.tau: int = 0  # elapsed time in step
        self.memory: np.ndarray = np.zeros(n_steps)  # base memory for each step
        self.global_tick: int = 0
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
        new_state.global_tick = self.global_tick
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

        # Cost structure (per-step failure costs + communication costs)
        c_fail_per_step: Optional[np.ndarray] = None,  # NEW: Per-step failure costs array
        c_remind: float = 1.0,             # Remind cost (assistant intervention)
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

    ):
        self.lambda_forget = lambda_forget
        self.delta_reminder = delta_reminder
        self.f0_base = f0_base
        self.k_memory = k_memory
        self.c_remind = c_remind
        self.c_nar = c_nar
        self.c_resp = c_resp
        self.c_off_timing = c_off_timing  # Penalty for off-timing reminders
        self.beta = beta
        self.obs_noise = obs_noise
        self.step_mean_duration = step_mean_duration
        self.step_std_duration = step_std_duration

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
            'c_remind': self.c_remind,
            'c_nar': self.c_nar,
            'c_resp': self.c_resp,
            'c_fail_per_step': self.c_fail_per_step.tolist() if isinstance(self.c_fail_per_step, np.ndarray) else self.c_fail_per_step,
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
        """Failure probability f(m) = f0 * exp(-k*m).

        Non-critical steps (criticality=0) always return 0.

        Examples:
        - m=0.0: f = 0.60 (60% failure, baseline)
        - m=0.6: f ≈ 0.07 (7% failure)
        - m=1.0: f ≈ 0.01 (1% failure)
        """
        if step_idx < len(self.task_def.steps):
            if self.task_def.steps[step_idx].criticality == 0:
                return 0.0

        memory = self.pa_state.memory[step_idx]
        return float(np.clip(self.params.f0_base * np.exp(-self.params.k_memory * memory), 0.0, 1.0))

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
        """Update memory based on assistant action.

        m_{n,t+1} = (1 - lambda) * m_{n,t} + Delta_A * I[a_t = remind_n]
        """
        self.pa_state.memory *= (1 - self.params.lambda_forget)

        for step_idx in range(self.n_steps):
            remind_key = f'remind_{step_idx}'
            if remind_key in self.assistant_actions:
                remind_action_id = self.assistant_actions[remind_key]
                if assistant_action == remind_action_id:
                    self.pa_state.memory[step_idx] += self.params.delta_reminder

        self.pa_state.memory = np.clip(self.pa_state.memory, 0, 2.0)
        self.pa_state.global_tick += 1

    def _sample_human_action(self, assistant_action: int, expected_value: float) -> int:
        """
        Human decides whether to communicate based on utility (Eqs. 5-6)
        P(narrate | silent) = sigma(beta * (g_t - c_nar))
        P(respond | confirm) = sigma(beta * (g_t - c_resp - c_remind))
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
            utility = expected_value - self.params.c_resp - self.params.c_remind
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
            reward -= self.params.c_remind
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
            # Action order: 0=silent, 1=confirm, 2..n+1=remind_i
            reminder_prob = 0.20 / n_steps if n_steps > 0 else 0.0
            self.action_probs = np.array([0.7, 0.1] + [reminder_prob] * n_steps)
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


if __name__ == "__main__":
    print("Procedure Assistant Simulation")
    print("=" * 60)
    print()

    # Test the environment
    print("Testing environment...")
    task_def = get_task_definition('make_cereal')
    params = SimulationParams()
    env = ProcedureAssistantEnv(params, task_def)
    obs = env.reset()
    print(f"Initial observation: step={obs['step_name']}, tau={obs['elapsed_time']}")
    print(f"Memory state: {obs['memory']}")
    print()
