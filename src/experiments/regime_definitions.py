"""Regime definitions for procedure assistant experiments.

Four orthogonal axes control the simulation:
- Failure cost regime: how expensive step failures are
- Communication cost regime: costs of narrate/remind/question/confirm
- Memory decay regime: forgetting rate and initial memory level
- Observation noise regime: narration effectiveness (noise baseline, floor, recovery rate)

Usage:
    from regime_definitions import build_params, COMM_COST_REGIMES
    params = build_params(task_def, 'balanced', 'cheap_narrate', 'step_transition', 'durable')
"""
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np

from task_definitions import TaskDefinition, create_per_step_failure_costs
from ma_procedure_assistant_sim import MASimulationParams


# ============================================================================
# COMMUNICATION COST REGIME
# ============================================================================

@dataclass(frozen=True)
class CommCostRegime:
    """Communication cost structure for both agents."""
    c_nar: float        # Human: narration cost
    c_remind: float     # Assistant: remind cost (was c_int)
    c_q: float          # Human: question cost
    c_confirm: float    # Assistant: confirm cost
    c_off_timing: float  # Assistant: penalty for remind at wrong step


COMM_COST_REGIMES: Dict[str, CommCostRegime] = {
    'default': CommCostRegime(
        c_nar=0.5, c_remind=1.0, c_q=0.5, c_confirm=1.0, c_off_timing=3.0,
    ),
    'cheap_narrate': CommCostRegime(
        c_nar=0.1, c_remind=1.0, c_q=0.5, c_confirm=0.5, c_off_timing=3.0,
    ),
}


# ============================================================================
# MEMORY DECAY REGIME
# ============================================================================

@dataclass(frozen=True)
class MemoryDecayRegime:
    """Memory dynamics parameters."""
    lambda_forget: float  # Forgetting rate (applied at step transitions)
    memory_init: float    # Initial memory level for all steps


MEMORY_DECAY_REGIMES: Dict[str, MemoryDecayRegime] = {
    'default': MemoryDecayRegime(lambda_forget=0.03, memory_init=0.0),
    # lambda_forget lowered 0.10 -> 0.03: prior value caused the joint policy to
    # spam reminders to outpace (1 - 0.10)^tau decay; 0.03 keeps memory durable
    # within a step so a single reminder suffices.
    'step_transition': MemoryDecayRegime(lambda_forget=0.03, memory_init=0.3),
}


# ============================================================================
# OBSERVATION NOISE REGIME
# ============================================================================

@dataclass(frozen=True)
class ObsNoiseRegime:
    """Observation noise dynamics — controls how effective narration is.

    Recovery formula: noise(t) = baseline - (baseline - min) * exp(-k * t)
    Half-life of narration effect: ln(2) / lambda_noise_recover ticks
    """
    obs_noise: float            # Baseline noise level (without narration)
    obs_noise_min: float        # Noise floor immediately after narration
    lambda_noise_recover: float  # Recovery rate; half-life = ln(2)/k ticks


OBS_NOISE_REGIMES: Dict[str, ObsNoiseRegime] = {
    # Half-life ≈ 7 ticks — effect fades within a single step (~30 ticks avg)
    'default': ObsNoiseRegime(
        obs_noise=0.5, obs_noise_min=0.05, lambda_noise_recover=0.10,
    ),
    # Half-life ≈ 35 ticks — effect persists across roughly one full step
    'durable': ObsNoiseRegime(
        obs_noise=0.5, obs_noise_min=0.05, lambda_noise_recover=0.02,
    ),
    # Low noise — assistant can observe human state more clearly
    'low_noise': ObsNoiseRegime(
        obs_noise=0.3, obs_noise_min=0.05, lambda_noise_recover=0.02,
    ),
    # High noise — assistant has poor observability
    'high_noise': ObsNoiseRegime(
        obs_noise=0.7, obs_noise_min=0.05, lambda_noise_recover=0.02,
    ),
    # Diagnostic: zero sensor noise. Assistant observes the true step
    # identity perfectly; narration's noise-reset effect is moot. Used to
    # isolate whether MA-IPPO's difficulty on long tasks is driven by
    # sensing fidelity (this regime should let it converge cleanly) vs.
    # something else (policy/reward shaping).
    'perfect': ObsNoiseRegime(
        obs_noise=0.0, obs_noise_min=0.0, lambda_noise_recover=0.02,
    ),
    # v6 accuracy-parametrized regimes (use with obs_noise_model='uniform').
    # obs_noise = 1 - baseline frame accuracy; obs_noise_min is the (small)
    # floor reached right after a narration/confirmation; durable persistence
    # (lambda 0.02). Sweep these to vary the per-frame step-recognizer accuracy.
    'acc100': ObsNoiseRegime(obs_noise=0.0,  obs_noise_min=0.0,  lambda_noise_recover=0.02),
    'acc90':  ObsNoiseRegime(obs_noise=0.1,  obs_noise_min=0.02, lambda_noise_recover=0.02),
    'acc75':  ObsNoiseRegime(obs_noise=0.25, obs_noise_min=0.02, lambda_noise_recover=0.02),
    'acc70':  ObsNoiseRegime(obs_noise=0.3,  obs_noise_min=0.02, lambda_noise_recover=0.02),
    'acc50':  ObsNoiseRegime(obs_noise=0.5,  obs_noise_min=0.02, lambda_noise_recover=0.02),
}


# ============================================================================
# FAILURE COST REGIME
# ============================================================================

FAILURE_COST_SCALES: Dict[str, float] = {
    'extremely_low': 2.0,
    'very_low': 5.0,
    'low': 10.0,
    'balanced': 15.0,
    'high': 30.0,
    'very_high': 50.0,
    'extremely_high': 50.0,  # kept for backward compatibility (alias for very_high)
}


# ============================================================================
# PARAMETER BUILDER
# ============================================================================

def build_params(
    task_def: TaskDefinition,
    fail_regime: str = 'balanced',
    comm_regime: str = 'default',
    decay_regime: str = 'default',
    obs_regime: str = 'default',
) -> MASimulationParams:
    """Build MASimulationParams from four orthogonal regime axes.

    Args:
        task_def: Task definition (determines n_steps, criticalities)
        fail_regime: Key into FAILURE_COST_SCALES
        comm_regime: Key into COMM_COST_REGIMES
        decay_regime: Key into MEMORY_DECAY_REGIMES
        obs_regime: Key into OBS_NOISE_REGIMES

    Returns:
        Fully configured MASimulationParams
    """
    comm = COMM_COST_REGIMES[comm_regime]
    decay = MEMORY_DECAY_REGIMES[decay_regime]
    obs = OBS_NOISE_REGIMES[obs_regime]
    c_fail_scale = FAILURE_COST_SCALES[fail_regime]
    c_fail_per_step = create_per_step_failure_costs(task_def, c_fail_scale)

    return MASimulationParams(
        # Memory dynamics
        lambda_forget=decay.lambda_forget,
        memory_init=decay.memory_init,
        delta_reminder=0.8,
        # Failure model
        f0_base=0.6,
        k_memory=3.0,
        c_fail_per_step=c_fail_per_step,
        # Communication costs
        c_remind=comm.c_remind,
        c_nar=comm.c_nar,
        c_q=comm.c_q,
        c_confirm=comm.c_confirm,
        c_off_timing=comm.c_off_timing,
        # Observation noise
        obs_noise=obs.obs_noise,
        obs_noise_min=obs.obs_noise_min,
        lambda_noise_recover=obs.lambda_noise_recover,
    )


def build_params_asymmetric(
    task_def: TaskDefinition,
    c_fail_scale: float,
    c_nar: float,
    c_remind: float,
    c_q: Optional[float] = None,
    c_confirm: Optional[float] = None,
    c_off_timing: float = 3.0,
    decay_regime: str = 'step_transition',
    obs_regime: str = 'durable',
) -> MASimulationParams:
    """Build MASimulationParams with independent human and assistant comm costs.

    The two ``axes'' of the cost-asymmetry phase diagram (E1):
    - c_nar (and by default c_q = c_nar) is the *human's* communication price.
    - c_remind (and by default c_confirm = c_remind) is the *assistant's*.

    Sweeping (c_nar, c_remind) independently exposes how the optimal division
    of communicative labor between the two agents shifts with cost asymmetry.

    Args:
        task_def: Task definition
        c_fail_scale: Failure cost scale (multiplier on per-step criticality)
        c_nar: Human narration cost
        c_remind: Assistant remind cost
        c_q: Human question cost (default: c_nar)
        c_confirm: Assistant confirm cost (default: c_remind)
        c_off_timing: Off-timing penalty for remind (default 3.0)
        decay_regime: Key into MEMORY_DECAY_REGIMES
        obs_regime: Key into OBS_NOISE_REGIMES

    Returns:
        Fully configured MASimulationParams
    """
    if c_q is None:
        c_q = c_nar
    if c_confirm is None:
        c_confirm = c_remind

    decay = MEMORY_DECAY_REGIMES[decay_regime]
    obs = OBS_NOISE_REGIMES[obs_regime]
    # v5: c_fail_scale is vestigial. Failure consequences are now fixed
    # constants on MASimulationParams (c_fail_critical=50 terminal,
    # c_fail_noncritical=1.0 soft); c_fail_per_step is no longer read by the
    # dynamics. We still build the array (and keep the --c-fail-scale CLI/
    # filename tag) for backward compatibility, but it has no effect.
    c_fail_per_step = create_per_step_failure_costs(task_def, c_fail_scale)

    return MASimulationParams(
        lambda_forget=decay.lambda_forget,
        memory_init=decay.memory_init,
        delta_reminder=0.8,
        f0_base=0.6,
        k_memory=3.0,
        c_fail_per_step=c_fail_per_step,
        c_remind=c_remind,
        c_nar=c_nar,
        c_q=c_q,
        c_confirm=c_confirm,
        c_off_timing=c_off_timing,
        obs_noise=obs.obs_noise,
        obs_noise_min=obs.obs_noise_min,
        lambda_noise_recover=obs.lambda_noise_recover,
    )


def build_params_sensing(
    task_def: TaskDefinition,
    c_fail_scale: float,
    c_comm: float,
    obs_noise: float,
    lambda_noise_recover: float,
    obs_noise_min: float = 0.05,
    c_remind: float = 1.0,
    c_confirm: float = 1.0,
    c_off_timing: float = 3.0,
    decay_regime: str = 'step_transition',
) -> MASimulationParams:
    """Build MASimulationParams with explicit sensing-fidelity parameters.

    Used for the sensing-fidelity sweep (E3): sweeps (obs_noise, lambda_noise_recover)
    at fixed communication and failure costs to expose how the framework
    reallocates communicative load when narration becomes brittle.

    Args:
        task_def: Task definition
        c_fail_scale: Failure cost scale
        c_comm: Symmetric comm cost (sets c_nar = c_q = c_comm)
        obs_noise: Baseline noise level (without recent narration)
        lambda_noise_recover: Recovery rate; higher = noise re-accumulates faster (brittle narration)
        obs_noise_min: Noise floor immediately after narration
        c_remind, c_confirm, c_off_timing: assistant cost terms (defaults match grid)
        decay_regime: Key into MEMORY_DECAY_REGIMES

    Returns:
        Fully configured MASimulationParams
    """
    decay = MEMORY_DECAY_REGIMES[decay_regime]
    c_fail_per_step = create_per_step_failure_costs(task_def, c_fail_scale)

    return MASimulationParams(
        lambda_forget=decay.lambda_forget,
        memory_init=decay.memory_init,
        delta_reminder=0.8,
        f0_base=0.6,
        k_memory=3.0,
        c_fail_per_step=c_fail_per_step,
        c_remind=c_remind,
        c_nar=c_comm,
        c_q=c_comm,
        c_confirm=c_confirm,
        c_off_timing=c_off_timing,
        obs_noise=obs_noise,
        obs_noise_min=obs_noise_min,
        lambda_noise_recover=lambda_noise_recover,
    )


def build_params_grid(
    task_def: TaskDefinition,
    c_fail_scale: float,
    c_comm: float,
    c_remind: float = 1.0,
    c_confirm: float = 1.0,
    c_off_timing: float = 3.0,
    decay_regime: str = 'step_transition',
    obs_regime: str = 'durable',
) -> MASimulationParams:
    """Build MASimulationParams for grid search with scalar comm cost.

    Sets c_nar = c_q = c_comm; c_remind and c_confirm are fixed separately.

    Args:
        task_def: Task definition
        c_fail_scale: Failure cost scale (multiplier on per-step criticality)
        c_comm: Communication cost applied to both c_nar and c_q
        c_remind: Remind cost (default 1.0, fixed in grid search)
        c_confirm: Confirm cost (default 1.0)
        c_off_timing: Off-timing penalty for remind (default 3.0)
        decay_regime: Key into MEMORY_DECAY_REGIMES
        obs_regime: Key into OBS_NOISE_REGIMES

    Returns:
        Fully configured MASimulationParams
    """
    decay = MEMORY_DECAY_REGIMES[decay_regime]
    obs = OBS_NOISE_REGIMES[obs_regime]
    c_fail_per_step = create_per_step_failure_costs(task_def, c_fail_scale)

    return MASimulationParams(
        lambda_forget=decay.lambda_forget,
        memory_init=decay.memory_init,
        delta_reminder=0.8,
        f0_base=0.6,
        k_memory=3.0,
        c_fail_per_step=c_fail_per_step,
        c_remind=c_remind,
        c_nar=c_comm,
        c_q=c_comm,
        c_confirm=c_confirm,
        c_off_timing=c_off_timing,
        obs_noise=obs.obs_noise,
        obs_noise_min=obs.obs_noise_min,
        lambda_noise_recover=obs.lambda_noise_recover,
    )
