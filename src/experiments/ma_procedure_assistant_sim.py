"""
Multi-Agent Procedure Assistant Simulation

Extends the single-agent POMDP formulation to a cooperative Dec-POMDP where
both the human and the assistant are RL agents.

Key changes from single-agent version:
- Human takes explicit actions: {silent, narrate, question_i for critical steps}
- Assistant actions: {silent, confirm, remind_i for critical steps}
- Shared cooperative reward (fully cooperative)
- obs_noise_state is now a dynamic state variable (modified by narration)
- Human observation includes own memory of current step (perfect self-knowledge)
- Task completion bonus incentivizes human to finish the task

Action symmetry:
  Human A_h = {silent=0, narrate=1, question_i=2+j (for j-th critical step)}
  Assistant A_a = {silent=0, confirm=1, remind_i=2+j (for j-th critical step)}
  Both have action size 2 + N_critical.

Future extension:
  Strong/Weak intervention levels for assistant (Phase 2).
"""

import random
import numpy as np
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional
import json

try:
    from task_definitions import TaskDefinition
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from task_definitions import TaskDefinition

try:
    from procedure_assistant_sim import (
        ProcedureAssistantState,
        SimulationParams,
    )
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from procedure_assistant_sim import (
        ProcedureAssistantState,
        SimulationParams,
    )


# ============================================================================
# HUMAN ACTIONS (multi-agent version)
# ============================================================================
# Note: question actions are added dynamically based on critical steps.
# Base human actions:
MA_HUMAN_ACTIONS_BASE = {
    'silent': 0,
    'narrate': 1,
    # question_i: 2+j for j-th critical step (built per-task)
}


# ============================================================================
# EXTENDED PARAMETERS
# ============================================================================
class MASimulationParams(SimulationParams):
    """Extended parameters for the multi-agent procedure assistant simulation.

    Adds parameters for:
    - Human question action (memory boost, cost)
    - Observation noise dynamics (narration effect and recovery)
    - Task completion bonus
    - Confirm action cost (separated from c_remind)
    """

    def __init__(
        self,
        # --- All existing SimulationParams args ---
        lambda_forget: float = 0.03,
        delta_reminder: float = 0.8,
        f0_base: float = 0.6,
        k_memory: float = 3.0,
        c_fail_per_step: Optional[np.ndarray] = None,
        c_remind: float = 1.0,
        c_nar: float = 1.0,
        c_resp: float = 2.0,
        c_off_timing: float = 0.5,
        c_fail_base: Optional[float] = None,
        step_failure_costs: Optional[Dict] = None,
        beta: float = 1.0,
        obs_noise: float = 0.2,
        step_mean_duration: int = 30,
        step_std_duration: int = 10,
        # --- New multi-agent parameters ---
        # Human question action
        delta_q: float = 0.4,           # Memory boost from human asking a question
        c_q: float = 0.5,               # Cost of human asking a question (fixed constant)

        # Confirm cost (separated from c_remind for assistant)
        c_confirm: float = 1.0,         # Cost of assistant's confirm action (fixed constant)

        # Observation noise dynamics
        obs_noise_min: float = 0.02,       # Noise floor immediately after narration
        lambda_noise_recover: float = 0.10,  # Recovery rate (half-life ~7 ticks)

        # Observation noise MODEL (v6). The per-frame step recognizer misreads
        # the true step identity with probability obs_noise_state (= 1 - frame
        # accuracy). What it outputs on a misread:
        #   'uniform'  : a step drawn uniformly from the OTHER N-1 steps
        #                (confusion-matrix style; decoupled from step ordering).
        #                Frame accuracy is then exactly (1 - obs_noise_state).
        #   'gaussian' : true_identity + N(0,1) rounded/clipped (legacy; biases
        #                errors toward index-adjacent steps).
        # The assistant's tracker (Bayesian belief over identity) integrates
        # noisy frames over time and is unchanged in structure; only the
        # per-frame emission + matching likelihood differ.
        obs_noise_model: str = 'uniform',

        # Task completion bonus
        R_complete: float = 10.0,  # Fixed constant: clear terminal signal for task completion

        # Initial memory level (partial procedure knowledge)
        memory_init: float = 0.0,  # 0 = no prior knowledge, 0.3 = partial recall

        # Memory floor: minimum retention level below which decay cannot fall.
        # Operationalizes Bahrick's permastore (1984): well-learned material
        # plateaus and resists further forgetting. Default = -1.0 (sentinel)
        # is resolved to memory_init below, so "initial value = floor".
        memory_floor: float = -1.0,

        # Mid-subtask interruption penalty (Adamczyk & Bailey 2004; Iqbal &
        # Horvitz 2007). Assistant actions issued mid-subtask (tau > tau_boundary)
        # incur an extra cost, reflecting the empirical finding that
        # interrupting at task boundaries is markedly cheaper than mid-subtask.
        # Magnitude 0.25 puts mid-subtask cost at ~1.5x boundary cost (when
        # c_remind = 0.5), within the 1.3-2x empirical range.
        c_mid_step: float = 0.25,
        tau_boundary: int = 5,  # protected boundary window (ticks after a step transition)

        # Logistic failure model parameters (Einstein & McDaniel 2005 PM
        # retrieval). When use_logistic_failure=True, replace exp(-k*m) with
        # f_0 * (1 - sigmoid(slope * (m - threshold))). Slope and threshold
        # are chosen so f(memory_init=0.3) ~= 0.30 (close to the prior
        # exponential f(0.3) ~= 0.24, while saturating less harshly at low m).
        use_logistic_failure: bool = True,
        failure_threshold: float = 0.3,
        failure_slope: float = 6.0,

        # Confirmation habituation (alert/alarm fatigue, Cvach 2012): the
        # cost of the k-th confirmation in an episode is
        # c_confirm * min(c_confirm_cap_factor, (1 + c_confirm_growth) ** (k - 1)),
        # so repeated confirmations escalate in cost up to a cap. Default
        # growth 0.3 -> the 5th confirm costs ~2.9x the first; cap 27 keeps
        # the cost bounded once growth would exceed it (~13th confirm),
        # preventing PPO value-function overflow when a random/exploring
        # policy fires 50+ confirms in one episode (which otherwise yields
        # 1.3^50 ~ 5e5 -> rewards in the 1e4-1e6 range). Set growth=0.0 for
        # a flat cost; cap_factor=inf to recover uncapped exponential growth.
        c_confirm_growth: float = 0.3,
        c_confirm_cap_factor: float = 27.0,

        # H6: remind habituation. Same formula as confirm habituation but
        # the counter is per-step (each critical step has its own count of
        # prior reminds this episode). The k-th remind for a given step
        # costs c_remind * min(c_remind_cap_factor, (1 + c_remind_growth) ** (k - 1)).
        # Default growth=2.0 -> 2nd remind for the same step costs 3x, 3rd
        # costs 9x. Cap factor 27 (= 3^3) keeps the 4th+ remind at the same
        # cost as the 3rd, preventing PPO value-function overflow during
        # random exploration on long tasks (a random policy can otherwise
        # fire 100+ reminds for the same step, producing 1e30+ rewards).
        # Set growth=0.0 to recover a flat cost; cap_factor=inf to recover
        # uncapped exponential growth.
        c_remind_growth: float = 2.0,
        c_remind_cap_factor: float = 27.0,

        # v4: per-step habituation for the human's narrate and question
        # actions, mirroring remind (H6). The k-th narrate/question for a
        # given step costs c × min(cap, (1 + growth)^(k-1)). Effect (obs-noise
        # reset for narrate, memory boost for question) now applies EVERY
        # time; only the cost escalates (the old once-per-step effect gating
        # is removed). Defaults match remind for symmetry across all four
        # communicative actions.
        c_nar_growth: float = 2.0,
        c_nar_cap_factor: float = 27.0,
        c_q_growth: float = 2.0,
        c_q_cap_factor: float = 27.0,

        # v4/v5: every step carries a memory-modulated failure risk.
        #  - critical step failure (v5): TERMINAL. The episode ends
        #    immediately with penalty c_fail_critical and NO completion bonus
        #    ("a mistake you must avoid"; default 50 = 5x R_complete).
        #  - non-critical step failure: soft, costs c_fail_noncritical (1.0)
        #    and the episode continues.
        # c_fail_scale is removed (v5): critical/non-critical costs are fixed
        # constants, not a swept multiplier.
        c_fail_noncritical: float = 1.0,
        c_fail_critical: float = 50.0,
    ):
        super().__init__(
            lambda_forget=lambda_forget,
            delta_reminder=delta_reminder,
            f0_base=f0_base,
            k_memory=k_memory,
            c_fail_per_step=c_fail_per_step,
            c_remind=c_remind,
            c_nar=c_nar,
            c_resp=c_resp,
            c_off_timing=c_off_timing,
            c_fail_base=c_fail_base,
            step_failure_costs=step_failure_costs,
            beta=beta,
            obs_noise=obs_noise,
            step_mean_duration=step_mean_duration,
            step_std_duration=step_std_duration,
        )
        # Human question
        self.delta_q = delta_q
        self.c_q = c_q

        # Confirm cost (assistant)
        self.c_confirm = c_confirm

        # Noise dynamics
        self.obs_noise_baseline = obs_noise   # obs_noise is now the baseline
        self.obs_noise_min = obs_noise_min
        self.lambda_noise_recover = lambda_noise_recover
        self.obs_noise_model = obs_noise_model

        # Completion bonus (fixed constant)
        self.R_complete = R_complete

        # Initial memory
        self.memory_init = memory_init

        # Memory floor (Bahrick permastore). Sentinel -1.0 means "use memory_init",
        # so the default is "initial value = floor".
        self.memory_floor = memory_init if memory_floor < 0 else memory_floor

        # Mid-subtask interruption penalty (Adamczyk-Bailey).
        self.c_mid_step = c_mid_step
        self.tau_boundary = tau_boundary

        # Logistic failure model.
        self.use_logistic_failure = use_logistic_failure
        self.failure_threshold = failure_threshold
        self.failure_slope = failure_slope

        # Confirmation habituation.
        self.c_confirm_growth = c_confirm_growth
        self.c_confirm_cap_factor = c_confirm_cap_factor

        # Remind habituation (H6).
        self.c_remind_growth = c_remind_growth
        self.c_remind_cap_factor = c_remind_cap_factor

        # v4: narrate/question habituation + non-critical failure cost.
        self.c_nar_growth = c_nar_growth
        self.c_nar_cap_factor = c_nar_cap_factor
        self.c_q_growth = c_q_growth
        self.c_q_cap_factor = c_q_cap_factor
        self.c_fail_noncritical = c_fail_noncritical
        self.c_fail_critical = c_fail_critical

    def resolve_R_complete(self, c_fail_per_step: np.ndarray) -> float:
        """Return the task completion bonus (fixed constant)."""
        return self.R_complete

    def to_dict(self):
        d = super().to_dict()
        d.update({
            'delta_q': self.delta_q,
            'c_q': self.c_q,
            'c_confirm': self.c_confirm,
            'obs_noise_min': self.obs_noise_min,
            'lambda_noise_recover': self.lambda_noise_recover,
            'obs_noise_model': self.obs_noise_model,
            'memory_init': self.memory_init,
            'memory_floor': self.memory_floor,
            'c_mid_step': self.c_mid_step,
            'tau_boundary': self.tau_boundary,
            'use_logistic_failure': self.use_logistic_failure,
            'failure_threshold': self.failure_threshold,
            'failure_slope': self.failure_slope,
            'c_confirm_growth': self.c_confirm_growth,
            'c_confirm_cap_factor': self.c_confirm_cap_factor,
            'c_remind_growth': self.c_remind_growth,
            'c_remind_cap_factor': self.c_remind_cap_factor,
            'c_nar_growth': self.c_nar_growth,
            'c_nar_cap_factor': self.c_nar_cap_factor,
            'c_q_growth': self.c_q_growth,
            'c_q_cap_factor': self.c_q_cap_factor,
            'c_fail_noncritical': self.c_fail_noncritical,
            'c_fail_critical': self.c_fail_critical,
        })
        return d


# ============================================================================
# EXTENDED STATE (v2: adds belief state for assistant)
# ============================================================================
class MAProcedureAssistantState(ProcedureAssistantState):
    """Extended state for the multi-agent environment (v3: semi-Markov belief).

    Adds (beyond ProcedureAssistantState):
    - obs_noise_state: current observation noise level (modified by narration)
    - ticks_since_narration: ticks since last human narration (for noise recovery)
    - human_last_action: action human took last tick
    - assistant_last_action: action assistant took last tick
    - total_questions: number of times human asked a question

    v3 belief state (PrISM-style semi-Markov POMDP):
    - step_tau_belief: joint distribution over (step identity, elapsed τ).
        shape (N+1, tau_max+1). Index N along axis 0 is the absorbing done state
        (stored at (N, 0); other τ columns remain 0).
    - memory_estimate_critical: assistant's estimate of memory for critical steps only
                                 computed from interaction history, NOT ground truth
    """

    def __init__(
        self,
        n_steps: int,
        n_critical: int,
        tau_max: int,
        obs_noise_baseline: float = 0.2,
    ):
        super().__init__(n_steps)
        self.obs_noise_state: float = obs_noise_baseline
        self.ticks_since_narration: int = 9999
        self.human_last_action: int = 0
        self.assistant_last_action: int = 0
        self.total_questions: int = 0
        self.total_reminds: int = 0
        self.total_confirms: int = 0

        # v4: per-step counters of prior narrate/question/remind this episode.
        # Drive per-step habituation: the k-th action targeting step s costs
        # c * min(cap, (1 + growth) ** (k - 1)). The communicative EFFECT
        # (obs-noise reset for narrate, memory boost for question/remind)
        # applies on EVERY attempt; only the cost escalates. (Replaces the
        # v3 once-per-step effect-gating sets narrated_steps / asked_steps.)
        self.narrated_step_count: dict = {}
        self.asked_step_count: dict = {}
        self.reminded_step_count: dict = {}

        # v3: Semi-Markov joint belief for assistant (N+1, tau_max+1)
        self.tau_max: int = tau_max
        self.step_tau_belief: np.ndarray = np.zeros((n_steps + 1, tau_max + 1))
        self.step_tau_belief[0, 0] = 1.0           # start certain at (step 0, τ=0)

        # v4: memory_estimate[i] = assistant's estimate of the human's memory
        # for step i (ALL steps, not just critical), built from interaction
        # history (remind + question + decay), NOT ground truth. Generalized
        # from the v3 critical-only estimate so the assistant can reason about
        # reminding any step.
        self.memory_estimate: np.ndarray = np.zeros(n_steps)

    @property
    def step_belief(self) -> np.ndarray:
        """Marginal belief over step identity (sum over τ). Shape (N+1,)."""
        return self.step_tau_belief.sum(axis=1)

    def copy(self):
        n_steps = len(self.memory)
        new_state = MAProcedureAssistantState(
            n_steps, len(self.memory_estimate), self.tau_max, self.obs_noise_state
        )
        # Copy base fields
        new_state.current_step = self.current_step
        new_state.tau = self.tau
        new_state.memory = self.memory.copy()
        new_state.global_tick = self.global_tick
        new_state.is_done = self.is_done
        new_state.total_failures = self.total_failures
        new_state.total_interactions = self.total_interactions
        new_state.total_responses = self.total_responses
        new_state.total_narrations = self.total_narrations
        # Copy MA-specific fields
        new_state.obs_noise_state = self.obs_noise_state
        new_state.ticks_since_narration = self.ticks_since_narration
        new_state.human_last_action = self.human_last_action
        new_state.assistant_last_action = self.assistant_last_action
        new_state.total_questions = self.total_questions
        new_state.total_reminds = self.total_reminds
        new_state.total_confirms = self.total_confirms
        new_state.narrated_step_count = dict(self.narrated_step_count)
        new_state.asked_step_count = dict(self.asked_step_count)
        new_state.reminded_step_count = dict(self.reminded_step_count)
        # Copy v3 belief fields
        new_state.step_tau_belief = self.step_tau_belief.copy()
        new_state.memory_estimate = self.memory_estimate.copy()
        return new_state


# ============================================================================
# MULTI-AGENT ENVIRONMENT
# ============================================================================
class MAProcedureAssistantEnv:
    """
    Cooperative Dec-POMDP: Procedure assistant with both human and assistant as RL agents.

    Both agents act simultaneously each tick (Parallel multi-agent).
    Reward is fully shared (cooperative game).

    Action spaces:
        Human:     {silent=0, narrate=1, question_j=2+j for j-th critical step}
        Assistant: {silent=0, confirm=1, remind_j=2+j   for j-th critical step}
        Both spaces have size = 2 + N_critical.

    Observations:
        Human (5 floats):
            [current_step, tau, memory[current_step], assistant_last_action, obs_noise_state]
        Assistant (semi-Markov belief, PrISM-style):
            step_belief:    marginal over identity (N+1,) — derived from joint
            expected_tau:   E[τ | s] per identity (N+1,) — dwell summary
            step_tau_belief: full joint (N+1, tau_max+1) — sufficient statistic
            memory_estimate_critical: estimated memory for critical steps from interaction history
                                      (NO ground truth access — only interactions are tracked)
            human_last_action: last action taken by the human

    Args:
        params: MASimulationParams
        task_def: TaskDefinition specifying the procedural task
    """

    def __init__(self, params: MASimulationParams, task_def: TaskDefinition):
        self.params = params
        self.task_def = task_def
        self.n_steps = task_def.n_steps
        self.procedural_steps = task_def.step_names

        # Identify critical steps (those with criticality > 0)
        self.critical_steps: List[int] = [
            i for i, step in enumerate(task_def.steps) if step.criticality > 0
        ]
        self.n_critical = len(self.critical_steps)

        # Build action spaces
        self.human_actions = self._build_human_action_space()
        self.assistant_actions = self._build_assistant_action_space()
        self.n_human_actions = len(self.human_actions)
        self.n_assistant_actions = len(self.assistant_actions)

        # v4: reverse map action_id → step_idx for remind actions. The
        # assistant can now remind about ANY step (not just critical), so the
        # map covers all N steps: action (2 + i) reminds step i.
        self._remind_id_to_step: Dict[int, int] = {
            2 + i: i for i in range(self.n_steps)
        }

        # Task completion bonus
        self.R_complete = params.resolve_R_complete(params.c_fail_per_step)

        # Stochastic ordering: identity transition matrix (averaged over valid orderings)
        self.identity_transition_matrix: np.ndarray = (
            self._compute_identity_transition_matrix()
        )
        # Current episode ordering (position → identity mapping, sampled each reset)
        self.episode_order: np.ndarray = np.arange(self.n_steps)

        # Semi-Markov dwell model (PrISM-style): per-step Gaussian survival.
        # tau_max bounds the joint-belief grid; saturating mass is held at tau_max.
        self.tau_max: int = int(np.ceil(max(
            s.mean_duration + 4.0 * max(s.std_duration, 1.0)
            for s in task_def.steps
        )))
        # step_dwell_escape[s, tau] = P(step s completes at tau+1 | survived to tau)
        self.step_dwell_escape: np.ndarray = self._compute_dwell_escape_matrix()

        # State
        self.ma_state = MAProcedureAssistantState(
            self.n_steps, self.n_critical, self.tau_max, params.obs_noise_baseline
        )

        # Step duration tracking
        self.step_target_duration: Dict[str, int] = {}
        self._sample_step_durations()

    def _compute_dwell_escape_matrix(self) -> np.ndarray:
        """Precompute escape(s, τ) = 1 − Φ̄(τ+1)/Φ̄(τ) from Gaussian dwell per step.

        Matches PrISM-Tracker's semi-Markov time-dependent self-loop escape.
        """
        N = self.n_steps
        esc = np.zeros((N, self.tau_max + 1))
        taus = np.arange(self.tau_max + 2)  # need τ and τ+1
        for s, step_def in enumerate(self.task_def.steps):
            mu = float(step_def.mean_duration)
            sigma = max(float(step_def.std_duration), 1.0)
            surv = np.maximum(1.0 - norm.cdf(taus, loc=mu, scale=sigma), 1e-10)
            esc_row = 1.0 - surv[1:] / surv[:-1]
            esc[s, :] = np.clip(esc_row, 0.0, 1.0)
        return esc

    # -----------------------------------------------------------------------
    # ACTION SPACE BUILDERS
    # -----------------------------------------------------------------------
    def _build_human_action_space(self) -> Dict[str, int]:
        """Build human action space — three actions independent of task size.

        Returns:
            Dict with keys: 'silent', 'narrate', 'question_next'.
            v4: `question_next` targets the NEXT step (current_step + 1)
            regardless of criticality, so the human's decision is purely
            "when to ask about what's coming". Per-step habituation (cost
            escalates for repeated questions about the same target) replaces
            the v3 ask-once gating.
        """
        return {'silent': 0, 'narrate': 1, 'question_next': 2}

    def _next_step(self, current_step: int):
        """The next POSITION (current_step + 1), or None if at/past the end."""
        nxt = current_step + 1
        return nxt if nxt < self.n_steps else None

    def _current_identity(self, current_step: int):
        """Identity performed at the given position, or None if at/past end."""
        if current_step < 0 or current_step >= self.n_steps:
            return None
        return int(self.episode_order[current_step])

    def _next_identity(self, current_step: int):
        """Identity performed at the NEXT position (current_step + 1), or None."""
        nxt = current_step + 1
        if nxt >= self.n_steps:
            return None
        return int(self.episode_order[nxt])

    def _remind_reach_distance(self, identity: int, position: int) -> int:
        """v5 path-aware off-timing distance.

        How many steps until `identity` is reached from `position`, taking the
        MINIMUM over all rollout patterns consistent with the realized prefix
        (episode_order[:position+1]). Reminding an identity that is upcoming
        soon on some feasible path is "well-timed" (small distance); one that
        is far ahead on every feasible path, or already passed, is "off-timed".

        Returns 0 if the identity is the one at the current position, the
        minimum positive look-ahead distance if it is upcoming, or n_steps
        (max, fully penalized) if no feasible path reaches it ahead.
        """
        eo = [int(x) for x in self.episode_order]
        if identity == eo[position]:
            return 0
        patterns = self.task_def.rollout_patterns or [eo]
        prefix = eo[:position + 1]
        best = None
        for pat in patterns:
            pat = [int(x) for x in pat]
            if pat[:position + 1] != prefix:
                continue  # not consistent with what has actually happened
            # find identity strictly ahead of the current position
            for q in range(position + 1, len(pat)):
                if pat[q] == identity:
                    d = q - position
                    if best is None or d < best:
                        best = d
                    break
        return best if best is not None else self.n_steps

    def _build_assistant_action_space(self) -> Dict[str, int]:
        """Build assistant action space.

        Returns:
            Dict with keys: 'silent', 'confirm', 'remind_<step_idx>'.
            v4: reminder actions target ANY step (action 2+i reminds step i),
            so the assistant must learn which steps are worth reminding rather
            than being handed the critical set in the action space.
        """
        actions = {'silent': 0, 'confirm': 1}
        for i in range(self.n_steps):
            actions[f'remind_{i}'] = 2 + i
        return actions

    # -----------------------------------------------------------------------
    # STOCHASTIC ORDERING
    # -----------------------------------------------------------------------
    def _sample_episode_order(self) -> np.ndarray:
        """Sample one rollout pattern uniformly from task_def.rollout_patterns.

        Returns:
            Array of length n_steps where array[position] = identity.
            If rollout_patterns is empty, returns the default identity ordering.
        """
        patterns = self.task_def.rollout_patterns
        if not patterns:
            return np.arange(self.n_steps, dtype=np.int32)
        chosen = random.choice(patterns)
        return np.array(chosen, dtype=np.int32)

    def _compute_identity_transition_matrix(self) -> np.ndarray:
        """Compute the expected identity transition matrix T averaged over rollout_patterns.

        T[i][j] = P(next identity = j | current identity = i), averaged uniformly
        over all patterns in task_def.rollout_patterns.

        Returns:
            T of shape (N+1, N+1) where N = n_steps and index N is the done state.
        """
        N = self.n_steps
        T = np.zeros((N + 1, N + 1))

        patterns = self.task_def.rollout_patterns
        if not patterns:
            # Fixed ordering: deterministic transitions
            for i in range(N):
                T[i][i + 1] = 1.0
            T[N][N] = 1.0
            return T

        n_patterns = len(patterns)
        for pattern in patterns:
            for pos in range(N):
                i = pattern[pos]
                j = pattern[pos + 1] if pos + 1 < N else N
                T[i][j] += 1.0 / n_patterns

        # Done state is absorbing
        T[N][N] = 1.0

        # Normalize rows for numerical safety
        for i in range(N):
            row_sum = T[i].sum()
            if row_sum > 1e-10:
                T[i] /= row_sum
        return T

    # -----------------------------------------------------------------------
    # STEP DURATION
    # -----------------------------------------------------------------------
    def _sample_step_durations(self):
        """Sample target durations for each step from truncated normal."""
        for step_def in self.task_def.steps:
            duration = max(5, int(np.random.normal(
                step_def.mean_duration,
                step_def.std_duration
            )))
            self.step_target_duration[step_def.name] = duration

    # -----------------------------------------------------------------------
    # RESET
    # -----------------------------------------------------------------------
    def reset(self) -> Tuple[Dict, Dict]:
        """Reset the environment.

        Samples a new episode ordering from swappable_groups and initializes the
        assistant's belief to a point mass at the identity of the first step.

        Returns:
            (human_obs, assistant_obs)
        """
        self.episode_order = self._sample_episode_order()
        self.ma_state = MAProcedureAssistantState(
            self.n_steps, self.n_critical, self.tau_max, self.params.obs_noise_baseline
        )
        # Set initial memory level (partial procedure knowledge)
        if self.params.memory_init > 0.0:
            self.ma_state.memory[:] = self.params.memory_init
        # Initialize joint belief at (identity of position 0, τ=0)
        initial_identity = int(self.episode_order[0])
        self.ma_state.step_tau_belief = np.zeros(
            (self.n_steps + 1, self.tau_max + 1)
        )
        self.ma_state.step_tau_belief[initial_identity, 0] = 1.0

        self._sample_step_durations()
        return self._get_human_observation(), self._get_assistant_observation()

    # -----------------------------------------------------------------------
    # OBSERVATIONS
    # -----------------------------------------------------------------------
    def _asked_count_vector(self) -> np.ndarray:
        """v4: per-step count of prior questions this episode (N-dim).
        Exposing the running count makes the per-step question habituation
        learnable (repeated questions about the same step cost more).
        """
        return np.array(
            [float(self.ma_state.asked_step_count.get(i, 0))
             for i in range(self.n_steps)],
            dtype=np.float32,
        )

    def _reminded_count_vector(self) -> np.ndarray:
        """v4: per-step count of prior reminds this episode (N-dim, H6
        habituation). Lets the assistant policy see that further reminds of a
        step are exponentially more expensive.
        """
        return np.array(
            [float(self.ma_state.reminded_step_count.get(i, 0))
             for i in range(self.n_steps)],
            dtype=np.float32,
        )

    def _get_human_observation(self) -> Dict:
        """Human's observation: perfect identity knowledge + own memory + last assistant action.

        With stochastic ordering, the human knows their current step IDENTITY
        (what they are actually performing), not their position in the sequence.

        Returns:
            Dict keys: current_identity, tau, memory_current, memory_next,
                       assistant_last_action, obs_noise_state,
                       narrated_current_count, asked_next_count, is_done
        """
        current_step = self.ma_state.current_step
        is_done = self.ma_state.is_done or current_step >= self.n_steps

        if is_done:
            return {
                'current_identity': self.n_steps,   # sentinel for done
                'tau': self.ma_state.tau,
                'memory_current': 0.0,
                'memory_next': 0.0,
                'assistant_last_action': self.ma_state.assistant_last_action,
                'obs_noise_state': self.ma_state.obs_noise_state,
                'narrated_current_count': 0.0,
                'asked_next_count': 0.0,
                'is_done': True,
            }

        # v5: memory and habituation counts are indexed by step IDENTITY.
        current_identity = int(self.episode_order[current_step])
        next_identity = self._next_identity(current_step)
        memory_current = float(self.ma_state.memory[current_identity])
        memory_next = (float(self.ma_state.memory[next_identity])
                       if next_identity is not None else 0.0)
        narrated_current_count = float(
            self.ma_state.narrated_step_count.get(current_identity, 0)
        )
        asked_next_count = float(
            self.ma_state.asked_step_count.get(next_identity, 0)
            if next_identity is not None else 0
        )
        return {
            'current_identity': current_identity,
            'tau': self.ma_state.tau,
            'memory_current': memory_current,
            'memory_next': memory_next,
            'assistant_last_action': self.ma_state.assistant_last_action,
            'obs_noise_state': self.ma_state.obs_noise_state,
            'narrated_current_count': narrated_current_count,
            'asked_next_count': asked_next_count,
            'is_done': False,
        }

    def _get_assistant_observation(self) -> Dict:
        """v3: Semi-Markov belief observation for the assistant (PrISM-style POMDP).

        The joint belief b(s, τ) is maintained internally. The observation exposes
        three derived views:
        - step_belief: marginal over step identity (N+1,)
        - expected_tau: E[τ | s] per identity (N+1,) — "how deep into step s we are"
        - step_tau_belief: full joint (N+1, tau_max+1) for policies that want it

        Returns:
            Dict with keys:
                step_belief: np.ndarray (N+1,)
                expected_tau: np.ndarray (N+1,)
                step_tau_belief: np.ndarray (N+1, tau_max+1)
                memory_estimate: np.ndarray (N,) — estimated memory for all steps (v4)
                human_last_action: int — action human took last tick
                asked_count: np.ndarray (N,) — #prior questions per step this episode (v4)
                reminded_count: np.ndarray (N,) — #prior reminds per step this episode (v4, H6)
                true_step: int — ground truth step (for debugging only)
                is_done: bool
        """
        state = self.ma_state
        true_step = state.current_step
        is_done = state.is_done or true_step >= self.n_steps

        b = state.step_tau_belief                     # (N+1, tau_max+1)
        step_marginal = b.sum(axis=1)                 # (N+1,)
        tau_range = np.arange(state.tau_max + 1, dtype=np.float64)
        row_mass = np.maximum(step_marginal, 1e-10)
        expected_tau = (b @ tau_range) / row_mass     # (N+1,)

        return {
            'step_belief': step_marginal,
            'expected_tau': expected_tau,
            'step_tau_belief': b.copy(),
            'memory_estimate': state.memory_estimate.copy(),
            'human_last_action': state.human_last_action,
            'asked_count': self._asked_count_vector(),
            'reminded_count': self._reminded_count_vector(),
            'true_step': true_step,
            'is_done': is_done,
        }

    # -----------------------------------------------------------------------
    # DYNAMICS
    # -----------------------------------------------------------------------
    def _compute_failure_probability(self, identity: int,
                                     memory_value: Optional[float] = None) -> float:
        """Failure probability as a function of memory at the given step IDENTITY.

        Logistic form (default): f(m) = f_0 * (1 - sigmoid(slope * (m - threshold))).
        Anchors to the prospective-memory retrieval-probability framework
        (Einstein & McDaniel 2005): retrieval succeeds with logistic
        probability in activation, and a failure event occurs iff retrieval
        fails. f_0 caps the unaided failure rate; threshold marks the
        activation level for 50% retrieval; slope sets transition sharpness.

        Legacy exponential form: f(m) = f_0 * exp(-k*m). Toggled by
        params.use_logistic_failure for backward compatibility.

        v5: ALL steps carry a memory-modulated failure risk, indexed by step
        IDENTITY (memory[identity]). What distinguishes critical from
        non-critical is the failure consequence (critical = terminal
        c_fail_critical; non-critical = soft c_fail_noncritical), not the
        probability.
        """
        # v5+ ML-1/T4 fix: optionally evaluate failure prob at an externally
        # provided memory value (e.g., the pre-action memory snapshot taken
        # before this tick's question/remind boosts). This prevents the
        # same-tick reminder from retroactively saving a failing step.
        memory = self.ma_state.memory[identity] if memory_value is None else float(memory_value)
        if self.params.use_logistic_failure:
            z = self.params.failure_slope * (memory - self.params.failure_threshold)
            p_retrieve = 1.0 / (1.0 + np.exp(-z))
            return float(self.params.f0_base * (1.0 - p_retrieve))
        return float(np.clip(self.params.f0_base * np.exp(-self.params.k_memory * memory), 0.0, 1.0))

    def _check_step_completion(self) -> bool:
        """Discrete hazard model for step completion."""
        current_identity = int(self.episode_order[self.ma_state.current_step])
        current_step_name = self.procedural_steps[current_identity]
        target_duration = self.step_target_duration[current_step_name]
        if self.ma_state.tau >= target_duration:
            hazard = 0.8
        else:
            hazard = 0.05 + 0.3 * (self.ma_state.tau / target_duration)
        return bool(np.random.random() < hazard)

    def _update_obs_noise(self, human_action: int):
        """Update observation noise state based on human action.

        v4: EVERY narration of the current step drops noise to obs_noise_min
        (the once-per-step effect gating is removed; instead repeated
        narrations of the same step escalate in COST via per-step
        habituation, handled in the reward block). When the human does not
        narrate, noise recovers exponentially toward obs_noise_baseline.
        """
        current_step = self.ma_state.current_step
        narrated_this_tick = (
            human_action == self.human_actions['narrate']
            and current_step < self.n_steps
        )
        if narrated_this_tick:
            # Immediate drop on every narration of the current step
            self.ma_state.obs_noise_state = self.params.obs_noise_min
            self.ma_state.ticks_since_narration = 0
        else:
            # Exponential recovery toward baseline (also applies to
            # redundant repeat-narrations that have no belief-side effect)
            self.ma_state.ticks_since_narration = min(
                self.ma_state.ticks_since_narration + 1, 9999
            )
            baseline = self.params.obs_noise_baseline
            minimum = self.params.obs_noise_min
            k = self.params.lambda_noise_recover
            t = self.ma_state.ticks_since_narration
            self.ma_state.obs_noise_state = float(
                baseline - (baseline - minimum) * np.exp(-k * t)
            )

    def _update_memory(self, human_action: int, assistant_action: int):
        """Update memory based on both agents' actions.

        v4:
        Human question_next → memory[next_step] += delta_q  (EVERY time)
        Assistant remind_i  → memory[i] += delta_reminder   (EVERY time)

        The once-per-step / ask-once effect gating is removed; repeated
        actions about the same step still boost memory but escalate in COST
        via per-step habituation (handled in the reward block).

        Memory decay is NOT applied here — it is applied as a batch at step
        transitions (see step()), making memory stable within a step.
        """
        # No per-tick decay — batch decay at step transitions

        # v5: Human question_next boosts memory of the NEXT identity
        # (episode_order[pos+1]), every time it is asked.
        if human_action == self.human_actions['question_next']:
            target = self._next_identity(self.ma_state.current_step)
            if target is not None:
                self.ma_state.memory[target] += self.params.delta_q

        # Assistant reminder: boost memory of the reminded identity (any step).
        # Action (2+i) targets identity i; memory is identity-indexed.
        if assistant_action in self._remind_id_to_step:
            identity = self._remind_id_to_step[assistant_action]
            self.ma_state.memory[identity] += self.params.delta_reminder

        # Clip
        self.ma_state.memory = np.clip(self.ma_state.memory, 0.0, 2.0)

        # Increment global tick
        self.ma_state.global_tick += 1

    # -----------------------------------------------------------------------
    # BELIEF STATE UPDATES (v2)
    # -----------------------------------------------------------------------
    def _update_step_belief_prior(self):
        """Semi-Markov prior propagation (PrISM-style).

        For each active (s, τ):
            - stay: (s, τ) → (s, τ+1) with prob 1 − escape(s, τ)
            - transition: (s, τ) → (s', 0) with prob escape(s, τ) · T[s, s']
        The done state (row N) is absorbing and held at (N, 0).
        Mass arriving past τ_max saturates into the τ_max column.
        """
        state = self.ma_state
        N = self.n_steps
        tau_max = state.tau_max
        T = self.identity_transition_matrix          # (N+1, N+1)
        esc = self.step_dwell_escape                  # (N, tau_max+1)

        b = state.step_tau_belief                     # (N+1, tau_max+1)
        active = b[:N, :]                             # (N, tau_max+1)

        stay_mass = active * (1.0 - esc)              # (N, tau_max+1)
        escape_mass = active * esc                    # (N, tau_max+1)
        escape_per_s = escape_mass.sum(axis=1)        # (N,)

        new_b = np.zeros_like(b)
        # Stay: shift τ → τ+1, saturate at τ_max
        new_b[:N, 1:] += stay_mass[:, :-1]
        new_b[:N, tau_max] += stay_mass[:, tau_max]
        # Transitions into (s', τ=0) from any active s via T
        # T[:N, :].T @ escape_per_s → contribution per destination s' ∈ {0..N}
        trans_contrib = T[:N, :].T @ escape_per_s     # (N+1,)
        new_b[:, 0] += trans_contrib
        # Done state is absorbing
        new_b[N, 0] += b[N, 0]

        total = new_b.sum()
        if total > 1e-10:
            new_b /= total
        state.step_tau_belief = new_b

    def _update_step_belief_narration(self, true_step: int):
        """Narration reveals identity exactly; τ conditional is preserved.

        Sets belief to P(τ | s = true_step) from the prior belief. Falls back
        to a point mass at τ=0 if the prior had zero mass on true_step.
        """
        state = self.ma_state
        b = state.step_tau_belief
        new_b = np.zeros_like(b)
        row = b[true_step, :]
        row_sum = row.sum()
        if row_sum > 1e-10:
            new_b[true_step, :] = row / row_sum
        else:
            # Prior was wrong about identity — conservative fallback
            new_b[true_step, 0] = 1.0
        state.step_tau_belief = new_b

    def _update_step_belief_observation(self, obs_step: int):
        """Bayesian update from a noisy per-frame identity observation.

        Likelihood depends only on s (not τ), and matches the per-frame
        emission model (params.obs_noise_model):
          'uniform' : P(obs=o | true=s) = (1-p)*[s==o] + p/(N-1)*[s!=o]
          'gaussian': P(obs=o | true=s) = (1-p)*[s==o] + p * N(s-o; 0, 1)
        Applied uniformly across the τ axis via broadcasting.
        """
        state = self.ma_state
        N = self.n_steps
        p_noise = state.obs_noise_state

        if self.params.obs_noise_model == 'gaussian':
            inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
            s_idx = np.arange(N + 1)
            diff = (obs_step - s_idx).astype(np.float64)
            gaussian_val = inv_sqrt_2pi * np.exp(-0.5 * diff * diff)
            likelihood = p_noise * gaussian_val
            likelihood[obs_step] += (1.0 - p_noise)
        else:
            # 'uniform' confusion: off-diagonal mass spread evenly over the
            # other N-1 real steps. (Done-state index N also carries the
            # off-diagonal value; it is pruned by the prior elsewhere.)
            n_other = max(N - 1, 1)
            likelihood = np.full(N + 1, p_noise / n_other, dtype=np.float64)
            likelihood[obs_step] = (1.0 - p_noise)

        new_b = state.step_tau_belief * likelihood[:, np.newaxis]
        total = new_b.sum()
        if total > 1e-10:
            new_b /= total
        else:
            # Fallback: uniform over active (s, τ=0)
            new_b = np.zeros_like(state.step_tau_belief)
            new_b[:N, 0] = 1.0 / N
        state.step_tau_belief = new_b

    def _update_memory_estimate(self, h_action: int, a_action: int):
        """Update assistant's estimate of per-step memory from interaction history.

        v5: tracks ALL steps by IDENTITY (N-dim memory_estimate), built from
        observed communicative actions only — no access to ground-truth memory.

        v5+ ML-1 fix: per-tick continuous decay, NOT step-boundary burst decay.
        The earlier step-boundary decay (decay_factor applied in step() when
        step_completed=True) leaked ground-truth step transitions into the
        assistant's observation via discrete jumps in memory_estimate. We now
        decay continuously each tick by (1 - lambda_forget); over an expected
        dwell of ~30 ticks this matches the ground-truth's per-step burst on
        average while removing the leakage.

        Updates:
        - Per-tick: memory_estimate *= (1 - lambda_forget) and floor (if > 0)
        - Human question_next observed → memory_estimate[next_identity] += delta_q
        - Assistant remind_i (own action) → memory_estimate[identity i] += delta_reminder
        """
        state = self.ma_state

        # ML-1 fix: continuous per-tick decay (no ground-truth transition signal).
        state.memory_estimate *= (1.0 - self.params.lambda_forget)
        if self.params.memory_floor > 0.0:
            np.maximum(
                state.memory_estimate, self.params.memory_floor,
                out=state.memory_estimate,
            )

        # Human question_next: boost estimate for the next identity
        # (episode_order[pos+1]), every time, consistent with ground truth.
        if h_action == self.human_actions['question_next']:
            target = self._next_identity(self.ma_state.current_step)
            if target is not None:
                state.memory_estimate[target] += self.params.delta_q

        # Assistant remind: boost estimate of the reminded identity (any step)
        if a_action in self._remind_id_to_step:
            identity = self._remind_id_to_step[a_action]
            state.memory_estimate[identity] += self.params.delta_reminder

        # Clip
        state.memory_estimate = np.clip(state.memory_estimate, 0.0, 2.0)

    # -----------------------------------------------------------------------
    # STEP (MAIN FUNCTION)
    # -----------------------------------------------------------------------
    def step(
        self,
        human_action: int,
        assistant_action: int,
    ) -> Tuple[Dict, Dict, float, bool, Dict]:
        """Execute one tick with simultaneous actions from both agents.

        Simultaneous action resolution order:
        1. Human narrate → obs_noise_state drops immediately
        2. Human question → memory[i] += delta_q
        3. Assistant remind → memory[i] += delta_reminder
        4. Step completion hazard (always runs)
        5. Memory decay

        Args:
            human_action:     Integer index into human_actions
            assistant_action: Integer index into assistant_actions

        Returns:
            (human_obs, assistant_obs, reward, done, info)
        """
        # Snapshots for tracking-accuracy metric (before any update in this tick).
        global_tick_pre = self.ma_state.global_tick
        is_done_pre = self.ma_state.is_done

        # 1. Sample noisy per-FRAME identity observation (using CURRENT noise,
        #    before this tick's narration). With prob current_noise the frame
        #    recognizer misreads; otherwise it reports the true step.
        true_step = self.ma_state.current_step
        true_identity = int(self.episode_order[true_step])
        current_noise = self.ma_state.obs_noise_state
        if np.random.random() < current_noise:
            if self.params.obs_noise_model == 'gaussian':
                # Legacy: error biased toward index-adjacent steps.
                noise_sample = np.random.normal(0, 1.0)
                obs_identity = int(np.clip(true_identity + noise_sample, 0, self.n_steps - 1))
            else:
                # 'uniform' (default): misread = a step drawn uniformly from
                # the OTHER N-1 steps (confusion-matrix off-diagonal = uniform;
                # decoupled from step ordering). Frame accuracy is exactly
                # (1 - current_noise).
                if self.n_steps > 1:
                    o = np.random.randint(0, self.n_steps - 1)
                    obs_identity = o if o < true_identity else o + 1
                else:
                    obs_identity = true_identity
        else:
            obs_identity = true_identity

        # 2. Update observation noise state (narration drops it; else exponential recovery)
        self._update_obs_noise(human_action)

        # 3. Update step belief (identity-based)
        if human_action == self.human_actions['narrate']:
            # Narration: human reveals their current identity → point mass reset
            self._update_step_belief_narration(true_identity)
        else:
            # No narration: prior propagation + Bayesian update from noisy identity obs
            self._update_step_belief_prior()
            self._update_step_belief_observation(obs_identity)

        # 3b. Tracking-accuracy probe: MAP identity vs. ground truth.
        #     Skip tick 0 (initial belief is trivially a point mass on truth)
        #     and any tick that started with the env already done.
        if global_tick_pre > 0 and not is_done_pre:
            map_identity = int(np.argmax(self.ma_state.step_tau_belief.sum(axis=1)))
            tracking_map_correct: Optional[int] = int(map_identity == true_identity)
        else:
            tracking_map_correct = None

        # 4. Update memory estimate (assistant's belief about critical step memory)
        self._update_memory_estimate(human_action, assistant_action)

        # T4 fix: snapshot ground-truth memory for the CURRENT identity BEFORE
        # this tick's question/remind boosts apply. The failure check below uses
        # this pre-action value, so a same-tick reminder cannot retroactively
        # save a step that's about to complete. The reminder still takes effect
        # on future ticks/steps via the boosted memory written by _update_memory.
        pre_step = self.ma_state.current_step  # capture before possible advance
        cur_id_pre = self._current_identity(pre_step)
        pre_action_memory_cur = (
            float(self.ma_state.memory[cur_id_pre])
            if cur_id_pre is not None else None
        )

        # 5. Update ground truth memory (question + remind, then decay)
        self._update_memory(human_action, assistant_action)

        # 6. Check step completion (always runs regardless of communication)
        step_completed = self._check_step_completion()

        # 7. Compute reward
        reward = 0.0
        failure_occurred = False

        # v5: failure is evaluated at the step IDENTITY being performed at the
        # current position (episode_order[pos]), not the raw position.
        cur_id = self._current_identity(self.ma_state.current_step)
        critical_terminal = False
        if step_completed and cur_id is not None:
            # T4 fix: use the pre-action memory snapshot (same identity).
            fail_prob = self._compute_failure_probability(
                cur_id, memory_value=pre_action_memory_cur,
            )
            failure_occurred = bool(np.random.random() < fail_prob)
            if failure_occurred:
                self.ma_state.total_failures += 1
                is_critical = self.task_def.steps[cur_id].criticality > 0
                if is_critical:
                    # v5: critical failure is TERMINAL — large penalty, no
                    # completion bonus. The episode ends immediately.
                    reward -= float(self.params.c_fail_critical)
                    critical_terminal = True
                else:
                    # Non-critical failure: soft, episode continues.
                    reward -= float(self.params.c_fail_noncritical)

        if critical_terminal:
            self.ma_state.is_done = True
            # Move all belief mass to absorbing done-state (N, τ=0)
            self.ma_state.step_tau_belief = np.zeros(
                (self.n_steps + 1, self.tau_max + 1)
            )
            self.ma_state.step_tau_belief[self.n_steps, 0] = 1.0
        elif step_completed:
            # Batch GROUND-TRUTH memory decay at step transition: decay by
            # tau+1 ticks. Floor at params.memory_floor (Bahrick 1984
            # permastore): decay cannot drop activation below the baseline
            # retention level.
            # ML-1 fix: memory_estimate is NOT decayed here — that would leak
            # the ground-truth transition into the assistant's observation.
            # memory_estimate gets continuous per-tick decay in
            # _update_memory_estimate instead.
            elapsed = self.ma_state.tau + 1  # include current tick
            decay_factor = (1.0 - self.params.lambda_forget) ** elapsed
            self.ma_state.memory *= decay_factor
            if self.params.memory_floor > 0.0:
                np.maximum(
                    self.ma_state.memory, self.params.memory_floor,
                    out=self.ma_state.memory,
                )

            # Advance step
            self.ma_state.current_step += 1
            self.ma_state.tau = 0

            if self.ma_state.current_step >= self.n_steps:
                self.ma_state.is_done = True
                reward += self.R_complete   # completion bonus
                # Move all belief mass to absorbing done-state (N, τ=0)
                self.ma_state.step_tau_belief = np.zeros(
                    (self.n_steps + 1, self.tau_max + 1)
                )
                self.ma_state.step_tau_belief[self.n_steps, 0] = 1.0
        else:
            self.ma_state.tau += 1

        # Communication costs
        # --- Human ---
        # v5: per-IDENTITY habituation. narrate targets the current identity,
        # question targets the next identity (episode_order[pos+1]). The k-th
        # action about a given target identity costs
        # c * min(cap, (1+growth)^(k-1)); the effect still applies every time
        # (handled in _update_memory / _update_obs_noise).
        if human_action == self.human_actions['narrate']:
            tgt = cur_id  # current identity (None only if past the end)
            if tgt is not None:
                prior = self.ma_state.narrated_step_count.get(tgt, 0)
                esc = min((1.0 + self.params.c_nar_growth) ** prior,
                          self.params.c_nar_cap_factor)
                reward -= self.params.c_nar * esc
                self.ma_state.narrated_step_count[tgt] = prior + 1
            else:
                reward -= self.params.c_nar
            self.ma_state.total_narrations += 1
        elif human_action == self.human_actions['question_next']:
            tgt = self._next_identity(pre_step)
            if tgt is not None:
                prior = self.ma_state.asked_step_count.get(tgt, 0)
                esc = min((1.0 + self.params.c_q_growth) ** prior,
                          self.params.c_q_cap_factor)
                reward -= self.params.c_q * esc
                self.ma_state.asked_step_count[tgt] = prior + 1
            else:
                reward -= self.params.c_q  # asking past the end: flat cost
            self.ma_state.total_questions += 1

        # --- Assistant ---
        assistant_acted = False
        if assistant_action == self.assistant_actions['confirm']:
            # H2: confirmation habituation (alert fatigue, Cvach 2012). The
            # k-th confirmation in an episode costs c_confirm*(1+growth)^(k-1),
            # so repeated confirmations escalate. total_confirms is the count
            # of *prior* confirmations this episode (incremented just below).
            confirm_escalation = (
                (1.0 + self.params.c_confirm_growth) ** self.ma_state.total_confirms
            )
            confirm_escalation = min(
                confirm_escalation, self.params.c_confirm_cap_factor
            )
            confirm_cost = self.params.c_confirm * confirm_escalation
            reward -= confirm_cost
            self.ma_state.total_interactions += 1
            self.ma_state.total_confirms += 1
            assistant_acted = True

            # H0: confirmation refreshes the assistant's tracking. The
            # assistant solicits a confirmation of the current step and the
            # human's response sharpens the assistant's identity belief —
            # the assistant-paid analogue of human narration. Mechanically
            # identical to the narration noise reset, but the assistant pays.
            self.ma_state.obs_noise_state = self.params.obs_noise_min
            self.ma_state.ticks_since_narration = 0
        elif assistant_action in self._remind_id_to_step:
            # H6: per-IDENTITY remind habituation. The k-th remind of a given
            # identity costs c_remind * min(cap, (1+growth)^(k-1)). Memory
            # boost (in _update_memory) and the off-timing penalty below apply
            # on every attempt.
            reminded_identity = self._remind_id_to_step[assistant_action]
            prior = self.ma_state.reminded_step_count.get(reminded_identity, 0)
            escalation = (1.0 + self.params.c_remind_growth) ** prior
            escalation = min(escalation, self.params.c_remind_cap_factor)
            remind_cost = self.params.c_remind * escalation
            reward -= remind_cost
            self.ma_state.reminded_step_count[reminded_identity] = prior + 1
            self.ma_state.total_interactions += 1
            self.ma_state.total_reminds += 1
            assistant_acted = True

            # H1: path-aware off-timing penalty (v5). Reminding an identity
            # that is upcoming soon on SOME feasible path (given the realized
            # prefix) is free; reminding one that is far ahead on every
            # feasible path, or already passed, is penalized by its minimum
            # remaining distance. Distance 0/1 (now / next) is penalty-free.
            d = self._remind_reach_distance(reminded_identity, pre_step)
            reward -= self.params.c_off_timing * max(0, d - 1)

        # Mid-subtask interruption penalty (Adamczyk & Bailey 2004):
        # assistant interventions delivered deep into a subtask are more
        # disruptive than those near a step boundary. Boundary-protected
        # window is the first tau_boundary ticks of a step (where tau<=bound).
        # Applies to confirm and remind; off-timing is handled separately.
        if (assistant_acted
                and self.params.c_mid_step > 0.0
                and self.ma_state.tau > self.params.tau_boundary):
            reward -= self.params.c_mid_step

        # Track last actions for next tick's observations
        self.ma_state.human_last_action = human_action
        self.ma_state.assistant_last_action = assistant_action

        # Build observations
        human_obs = self._get_human_observation()
        assistant_obs = self._get_assistant_observation()

        done = self.ma_state.is_done
        info = {
            'failure': failure_occurred,
            'step_completed': step_completed,
            'human_action': human_action,
            'assistant_action': assistant_action,
            'true_step': self.ma_state.current_step,
            'obs_noise_state': self.ma_state.obs_noise_state,
            'memory': self.ma_state.memory.copy(),
            'tracking_map_correct': tracking_map_correct,
        }

        return human_obs, assistant_obs, reward, done, info

    # -----------------------------------------------------------------------
    # UTILITIES
    # -----------------------------------------------------------------------
    def get_action_names(self) -> Tuple[Dict[int, str], Dict[int, str]]:
        """Return (id→name) maps for human and assistant actions."""
        human_inv = {v: k for k, v in self.human_actions.items()}
        assistant_inv = {v: k for k, v in self.assistant_actions.items()}
        return human_inv, assistant_inv

    def get_state_summary(self) -> Dict:
        """Return a summary of the current state (for logging/debugging)."""
        return {
            'current_step': self.ma_state.current_step,
            'tau': self.ma_state.tau,
            'global_tick': self.ma_state.global_tick,
            'memory': self.ma_state.memory.tolist(),
            'obs_noise_state': self.ma_state.obs_noise_state,
            'ticks_since_narration': self.ma_state.ticks_since_narration,
            'total_failures': self.ma_state.total_failures,
            'total_narrations': self.ma_state.total_narrations,
            'total_questions': self.ma_state.total_questions,
            'total_interactions': self.ma_state.total_interactions,
            'total_reminds': self.ma_state.total_reminds,
            'total_confirms': self.ma_state.total_confirms,
            'is_done': self.ma_state.is_done,
        }


# ============================================================================
# BASELINE POLICIES
# ============================================================================
class RandomHumanPolicy:
    """Human baseline: uniformly random over actions."""

    def __init__(self, n_human_actions: int):
        self.n_actions = n_human_actions

    def get_action(self, obs: Dict) -> int:
        return int(np.random.randint(0, self.n_actions))

    def predict(self, obs: np.ndarray):
        """SB3-compatible predict interface."""
        return np.array(np.random.randint(0, self.n_actions)), None


class PassiveHumanPolicy:
    """Human baseline: always silent (never communicates).

    Equivalent to the passive human in the single-agent version.
    """

    def get_action(self, obs: Dict) -> int:
        return 0  # always silent

    def predict(self, obs: np.ndarray):
        return np.array(0), None


class AlwaysNarrateHumanPolicy:
    """Human baseline: always narrate.

    Upper bound on human communication (provides obs_noise reduction every tick).
    """

    def get_action(self, obs: Dict) -> int:
        return 1  # always narrate

    def predict(self, obs: np.ndarray):
        return np.array(1), None


class RandomAssistantPolicy:
    """Assistant baseline: uniformly random over actions."""

    def __init__(self, n_assistant_actions: int):
        self.n_actions = n_assistant_actions

    def get_action(self, obs: Dict) -> int:
        return int(np.random.randint(0, self.n_actions))

    def predict(self, obs: np.ndarray):
        return np.array(np.random.randint(0, self.n_actions)), None


class SilentAssistantPolicy:
    """Assistant baseline: always silent."""

    def get_action(self, obs: Dict) -> int:
        return 0

    def predict(self, obs: np.ndarray):
        return np.array(0), None


class HeuristicReminderAssistantPolicy:
    """Heuristic assistant: remind once when step_belief concentrates on a critical step.

    Uses the assistant's belief state (step_belief) to detect when the human
    is likely at a critical step, then issues a single remind for that step.
    Does not use confirm actions.

    Args:
        n_steps: Number of steps in the task
        critical_steps: List of step indices that are critical
        threshold: Belief threshold to trigger a remind (default: 0.3)
    """

    def __init__(self, n_steps: int, critical_steps: List[int], threshold: float = 0.3):
        self.n_steps = n_steps
        self.critical_steps = critical_steps
        self.threshold = threshold
        # v4: remind action id for step i is (2 + i) over ALL steps. The
        # heuristic still chooses to remind only critical steps (it uses
        # domain knowledge of the critical set), but maps to the new ids.
        self._step_to_remind_action: Dict[int, int] = {
            step_idx: 2 + step_idx for step_idx in critical_steps
        }
        self._reminded_steps: set = set()

    def reset(self):
        """Reset per-episode tracking. Must be called at the start of each episode."""
        self._reminded_steps = set()

    def get_action(self, obs: Dict) -> int:
        """Select action from dict-based observation."""
        step_belief = obs['step_belief']
        return self._select_action(step_belief)

    def predict(self, obs: np.ndarray):
        """SB3-compatible predict interface (flat array observation).

        Observation layout (v3):
          [step_belief(N+1), expected_tau(N+1), memory_est_critical(Nc), human_last_action(1)]
        Heuristic only needs the marginal step_belief at the front.
        """
        step_belief = obs[:self.n_steps + 1]
        return np.array(self._select_action(step_belief)), None

    def _select_action(self, step_belief: np.ndarray) -> int:
        """Core logic: remind once per critical step when belief exceeds threshold."""
        for step_idx in self.critical_steps:
            if step_idx in self._reminded_steps:
                continue
            if step_idx < len(step_belief) and step_belief[step_idx] >= self.threshold:
                self._reminded_steps.add(step_idx)
                return self._step_to_remind_action[step_idx]
        return 0  # silent
