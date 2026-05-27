"""
Comparison Experiment: None vs Passive Assistant vs Heuristic vs MA-IPPO

Sweeps across obs_noise levels, failure cost scales, and all 7 tasks to produce
a comprehensive empirical evaluation for the paper.

Four policies:
  1. None:              PassiveHumanPolicy + SilentAssistantPolicy  (zero communication floor)
  2. Passive Assistant: AlwaysNarrateHumanPolicy + SilentAssistantPolicy  (human-led only)
  3. Heuristic:         PassiveHumanPolicy + HeuristicReminderAssistantPolicy  (assistant-led, rule-based)
  4. MA-IPPO:           Trained human + assistant models  (proposed)

Usage:
    cd src/experiments
    python run_comparison_experiment.py --task make_cereal --obs-noise normal --fail-regime balanced
    python run_comparison_experiment.py --all --train-missing --eval-episodes 500

Output:
    data/results/comparison_4policy.json
"""

import sys
import json
import argparse
import time
import logging
import datetime
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training"))

from task_definitions import load_task_definitions, TaskDefinition
from ma_procedure_assistant_sim import (
    MAProcedureAssistantEnv,
    MASimulationParams,
    PassiveHumanPolicy,
    AlwaysNarrateHumanPolicy,
    SilentAssistantPolicy,
    HeuristicReminderAssistantPolicy,
)
from regime_definitions import (
    build_params,
    FAILURE_COST_SCALES,
    OBS_NOISE_REGIMES,
)

try:
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

from train_ma_ippo import (
    HumanGymWrapper,
    AssistantGymWrapper,
    train_ippo,
    evaluate_joint,
)

# ============================================================================
# CONSTANTS
# ============================================================================
TASKS = [
    'make_cereal', 'make_coffee', 'make_tea', 'make_sandwich',
    'cooking', 'make_stencil', 'latte_making',
]

OBS_NOISE_LEVELS = {
    'high': 'high_noise',
    'normal': 'durable',
    'low': 'low_noise',
}

FAIL_REGIMES = ['very_low', 'low', 'balanced', 'high', 'very_high']

# Default training settings
COMM_REGIME = 'cheap_narrate'
DECAY_REGIME = 'step_transition'
TRAIN_ROUNDS = 15
TRAIN_STEPS = 50_000


# ============================================================================
# MODEL DIRECTORY CONVENTION
# ============================================================================
def get_model_dir(
    task_name: str,
    fail_regime: str,
    obs_noise_key: str,
    suffix: Optional[str] = None,
) -> Path:
    """Get the model directory for a given condition.

    Normal noise (durable): models/ma_ippo[_<suffix>]/<task>/<fail_regime>/
    High/low noise:         models/ma_ippo[_<suffix>]/<task>/<fail_regime>_<obs_regime>/
    """
    root_name = "ma_ippo" if not suffix else f"ma_ippo_{suffix}"
    base = PROJECT_ROOT / "models" / root_name / task_name
    if obs_noise_key == 'normal':
        return base / fail_regime
    else:
        obs_regime = OBS_NOISE_LEVELS[obs_noise_key]
        return base / f"{fail_regime}_{obs_regime}"


def load_trained_models(
    model_dir: Path,
) -> Optional[Tuple[Any, Any]]:
    """Load trained human and assistant models from a directory.

    Tries best models first, then falls back to final models.

    Returns:
        (human_model, assistant_model) or None if models not found.
    """
    if not HAS_SB3:
        return None

    # Try best models first
    h_best = model_dir / "human_model_best.zip"
    a_best = model_dir / "assistant_model_best.zip"
    h_final = model_dir / "human_model_final.zip"
    a_final = model_dir / "assistant_model_final.zip"

    if h_best.exists() and a_best.exists():
        human_model = PPO.load(str(h_best))
        assistant_model = PPO.load(str(a_best))
        return human_model, assistant_model
    elif h_final.exists() and a_final.exists():
        human_model = PPO.load(str(h_final))
        assistant_model = PPO.load(str(a_final))
        return human_model, assistant_model

    return None


# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure file + console logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f"comparison_4policy_{timestamp}.log"

    logger = logging.getLogger('comparison_4policy')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

    logger.info(f"Log file: {log_path}")
    return logger


LOG = logging.getLogger('comparison_4policy')


# ============================================================================
# PER-EPISODE DATA COLLECTION
# ============================================================================
_EPISODE_KEYS = ['reward', 'failures', 'narrations', 'questions',
                 'reminds', 'confirms', 'ticks', 'tracking_map_acc']


def _summarize_episodes(episodes: List[Dict[str, float]]) -> Dict[str, Any]:
    """Compute summary stats from per-episode records and attach raw data."""
    arrays = {k: np.array([ep[k] for ep in episodes], dtype=float) for k in _EPISODE_KEYS}
    summary = {}
    for k, arr in arrays.items():
        summary[f'mean_{k}'] = float(np.nanmean(arr))
        summary[f'std_{k}'] = float(np.nanstd(arr))
        summary[f'median_{k}'] = float(np.nanmedian(arr))
        summary[f'min_{k}'] = float(np.nanmin(arr))
        summary[f'max_{k}'] = float(np.nanmax(arr))
    # Also store full per-episode data for later analysis
    summary['per_episode'] = [
        {k: float(ep[k]) for k in _EPISODE_KEYS} for ep in episodes
    ]
    summary['n_episodes'] = len(episodes)
    return summary


def _collect_episode(
    state, ep_reward: float, ep_tracking: List[int]
) -> Dict[str, float]:
    """Extract one episode record from the environment state.

    ep_tracking: list of per-tick `tracking_map_correct` ints (0/1), already
    filtered to exclude tick 0 and post-done ticks.
    """
    return {
        'reward': ep_reward,
        'failures': state.total_failures,
        'narrations': state.total_narrations,
        'questions': state.total_questions,
        'reminds': state.total_reminds,
        'confirms': state.total_confirms,
        'ticks': state.global_tick,
        'tracking_map_acc': (float(np.mean(ep_tracking)) if ep_tracking
                             else float('nan')),
    }


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================
def evaluate_none(
    ma_env: MAProcedureAssistantEnv,
    n_episodes: int = 500,
) -> Dict[str, Any]:
    """Evaluate "None" baseline: silent human + silent assistant.

    Zero communication floor. Both agents stay silent for the full episode,
    so failure risk is determined entirely by the un-aided memory trajectory.
    """
    human_policy = PassiveHumanPolicy()
    assistant_policy = SilentAssistantPolicy()

    episodes = []
    for _ in range(n_episodes):
        h_obs_dict, a_obs_dict = ma_env.reset()
        ep_reward = 0.0
        ep_tracking: List[int] = []
        done = False

        while not done:
            h_act = human_policy.get_action(h_obs_dict)
            a_act = assistant_policy.get_action(a_obs_dict)
            h_obs_dict, a_obs_dict, reward, done, info = ma_env.step(h_act, a_act)
            ep_reward += reward
            mc = info.get('tracking_map_correct')
            if mc is not None:
                ep_tracking.append(mc)

        episodes.append(_collect_episode(ma_env.ma_state, ep_reward, ep_tracking))

    return _summarize_episodes(episodes)


def evaluate_passive_assistant(
    ma_env: MAProcedureAssistantEnv,
    n_episodes: int = 500,
) -> Dict[str, Any]:
    """Evaluate "Passive Assistant" baseline: always-narrating human + silent assistant.

    The human narrates every tick (upper bound on human-side effort, providing
    constant obs-noise reduction); the assistant never intervenes. This isolates
    the human-led-only regime.
    """
    human_policy = AlwaysNarrateHumanPolicy()
    assistant_policy = SilentAssistantPolicy()

    episodes = []
    for _ in range(n_episodes):
        h_obs_dict, a_obs_dict = ma_env.reset()
        ep_reward = 0.0
        ep_tracking: List[int] = []
        done = False

        while not done:
            h_act = human_policy.get_action(h_obs_dict)
            a_act = assistant_policy.get_action(a_obs_dict)
            h_obs_dict, a_obs_dict, reward, done, info = ma_env.step(h_act, a_act)
            ep_reward += reward
            mc = info.get('tracking_map_correct')
            if mc is not None:
                ep_tracking.append(mc)

        episodes.append(_collect_episode(ma_env.ma_state, ep_reward, ep_tracking))

    return _summarize_episodes(episodes)


def evaluate_heuristic(
    ma_env: MAProcedureAssistantEnv,
    n_episodes: int = 500,
) -> Dict[str, Any]:
    """Evaluate "Heuristic" baseline: silent human + belief-thresholded reminder assistant.

    The assistant monitors the shared `step_belief` and, for each critical step
    in order, issues a single `remind_i` action the first time its belief mass
    exceeds 0.3; otherwise stays silent. The human stays silent.
    """
    human_policy = PassiveHumanPolicy()
    assistant_policy = HeuristicReminderAssistantPolicy(
        n_steps=ma_env.n_steps,
        critical_steps=ma_env.critical_steps,
        threshold=0.3,
    )

    episodes = []
    for _ in range(n_episodes):
        assistant_policy.reset()
        h_obs_dict, a_obs_dict = ma_env.reset()
        ep_reward = 0.0
        ep_tracking: List[int] = []
        done = False

        while not done:
            h_act = human_policy.get_action(h_obs_dict)
            a_act = assistant_policy.get_action(a_obs_dict)
            h_obs_dict, a_obs_dict, reward, done, info = ma_env.step(h_act, a_act)
            ep_reward += reward
            mc = info.get('tracking_map_correct')
            if mc is not None:
                ep_tracking.append(mc)

        episodes.append(_collect_episode(ma_env.ma_state, ep_reward, ep_tracking))

    return _summarize_episodes(episodes)


def evaluate_ma_ippo(
    ma_env: MAProcedureAssistantEnv,
    human_model,
    assistant_model,
    n_episodes: int = 500,
) -> Dict[str, Any]:
    """Evaluate trained MA-IPPO models with per-episode logging."""
    wrapper_h = HumanGymWrapper(ma_env, assistant_model)
    wrapper_a = AssistantGymWrapper(ma_env, human_model)

    episodes = []
    for _ in range(n_episodes):
        h_obs_dict, a_obs_dict = ma_env.reset()
        h_obs = wrapper_h._convert_human_obs(h_obs_dict)
        a_obs = wrapper_a._convert_assistant_obs(a_obs_dict)
        ep_reward = 0.0
        ep_tracking: List[int] = []
        done = False

        while not done:
            h_action, _ = human_model.predict(h_obs, deterministic=True)
            a_action, _ = assistant_model.predict(a_obs, deterministic=True)
            h_obs_dict, a_obs_dict, reward, done, info = ma_env.step(
                int(h_action), int(a_action)
            )
            h_obs = wrapper_h._convert_human_obs(h_obs_dict)
            a_obs = wrapper_a._convert_assistant_obs(a_obs_dict)
            ep_reward += reward
            mc = info.get('tracking_map_correct')
            if mc is not None:
                ep_tracking.append(mc)

        episodes.append(_collect_episode(ma_env.ma_state, ep_reward, ep_tracking))

    return _summarize_episodes(episodes)


# ============================================================================
# SINGLE CONDITION RUNNER
# ============================================================================
def run_single_condition(
    task_name: str,
    task_def: TaskDefinition,
    obs_noise_key: str,
    fail_regime: str,
    n_eval_episodes: int = 500,
    train_missing: bool = False,
    model_suffix: Optional[str] = None,
    train_rounds: int = TRAIN_ROUNDS,
    train_steps: int = TRAIN_STEPS,
) -> Dict[str, Any]:
    """Run evaluation for a single (task, obs_noise, fail_regime) condition.

    Args:
        task_name: Task identifier
        task_def: Task definition object
        obs_noise_key: 'high', 'normal', or 'low'
        fail_regime: One of FAIL_REGIMES
        n_eval_episodes: Number of episodes per policy
        train_missing: If True, train MA-IPPO models when not found

    Returns:
        Dict with results for all 3 policies
    """
    obs_regime = OBS_NOISE_LEVELS[obs_noise_key]
    params = build_params(task_def, fail_regime, COMM_REGIME, DECAY_REGIME, obs_regime)
    ma_env = MAProcedureAssistantEnv(params, task_def)

    condition_result: Dict[str, Any] = {
        'metadata': {
            'task': task_name,
            'obs_noise': obs_noise_key,
            'obs_regime': obs_regime,
            'fail_regime': fail_regime,
            'fail_cost_scale': FAILURE_COST_SCALES[fail_regime],
            'comm_regime': COMM_REGIME,
            'decay_regime': DECAY_REGIME,
            'n_steps': task_def.n_steps,
            'n_critical': ma_env.n_critical,
            'critical_steps': ma_env.critical_steps,
            'R_complete': float(ma_env.R_complete),
            'n_eval_episodes': n_eval_episodes,
            'params': params.to_dict(),
        }
    }

    # 1. None baseline (zero-communication floor)
    LOG.info(f"  [none] Evaluating {n_eval_episodes} episodes...")
    t0 = time.time()
    condition_result['none'] = evaluate_none(ma_env, n_eval_episodes)
    elapsed = time.time() - t0
    condition_result['none']['eval_time_s'] = elapsed
    LOG.info(f"  [none] {elapsed:.1f}s | "
             f"reward={condition_result['none']['mean_reward']:.2f} "
             f"(std={condition_result['none']['std_reward']:.2f}) "
             f"failures={condition_result['none']['mean_failures']:.2f} "
             f"ticks={condition_result['none']['mean_ticks']:.1f}")

    # 2. Passive Assistant baseline (human-led only)
    LOG.info(f"  [passive_assistant] Evaluating {n_eval_episodes} episodes...")
    t0 = time.time()
    condition_result['passive_assistant'] = evaluate_passive_assistant(ma_env, n_eval_episodes)
    elapsed = time.time() - t0
    condition_result['passive_assistant']['eval_time_s'] = elapsed
    LOG.info(f"  [passive_assistant] {elapsed:.1f}s | "
             f"reward={condition_result['passive_assistant']['mean_reward']:.2f} "
             f"(std={condition_result['passive_assistant']['std_reward']:.2f}) "
             f"failures={condition_result['passive_assistant']['mean_failures']:.2f} "
             f"narrations={condition_result['passive_assistant']['mean_narrations']:.2f} "
             f"ticks={condition_result['passive_assistant']['mean_ticks']:.1f}")

    # 3. Heuristic baseline (rule-based assistant-led)
    LOG.info(f"  [heuristic] Evaluating {n_eval_episodes} episodes...")
    t0 = time.time()
    condition_result['heuristic'] = evaluate_heuristic(ma_env, n_eval_episodes)
    elapsed = time.time() - t0
    condition_result['heuristic']['eval_time_s'] = elapsed
    LOG.info(f"  [heuristic] {elapsed:.1f}s | "
             f"reward={condition_result['heuristic']['mean_reward']:.2f} "
             f"(std={condition_result['heuristic']['std_reward']:.2f}) "
             f"failures={condition_result['heuristic']['mean_failures']:.2f} "
             f"reminds={condition_result['heuristic']['mean_reminds']:.2f} "
             f"ticks={condition_result['heuristic']['mean_ticks']:.1f}")

    # 4. MA-IPPO
    model_dir = get_model_dir(task_name, fail_regime, obs_noise_key, suffix=model_suffix)
    models = load_trained_models(model_dir)

    if models is None and train_missing:
        LOG.info(f"  [ma_ippo] Training ({train_rounds} rounds x {train_steps} steps) ...")
        t0 = time.time()
        human_model, assistant_model, training_log = train_ippo(
            params=params,
            task_def=task_def,
            n_rounds=train_rounds,
            steps_per_round=train_steps,
            save_dir=model_dir,
            verbose=1,
        )
        train_elapsed = time.time() - t0
        LOG.info(f"  [ma_ippo] Training done in {train_elapsed:.1f}s, "
                 f"saved to {model_dir}")
        models = (human_model, assistant_model)
        condition_result['ma_ippo_training'] = {
            'train_time_s': train_elapsed,
            'model_dir': str(model_dir),
            'rounds': training_log.get('rounds', []),
        }

    if models is not None:
        human_model, assistant_model = models
        LOG.info(f"  [ma_ippo] Evaluating {n_eval_episodes} episodes "
                 f"(model: {model_dir}) ...")
        t0 = time.time()
        condition_result['ma_ippo'] = evaluate_ma_ippo(
            ma_env, human_model, assistant_model, n_eval_episodes
        )
        elapsed = time.time() - t0
        condition_result['ma_ippo']['eval_time_s'] = elapsed
        condition_result['ma_ippo']['model_dir'] = str(model_dir)
        LOG.info(f"  [ma_ippo] {elapsed:.1f}s | "
                 f"reward={condition_result['ma_ippo']['mean_reward']:.2f} "
                 f"(std={condition_result['ma_ippo']['std_reward']:.2f}) "
                 f"failures={condition_result['ma_ippo']['mean_failures']:.2f} "
                 f"narrations={condition_result['ma_ippo']['mean_narrations']:.2f} "
                 f"questions={condition_result['ma_ippo']['mean_questions']:.2f} "
                 f"reminds={condition_result['ma_ippo']['mean_reminds']:.2f} "
                 f"confirms={condition_result['ma_ippo']['mean_confirms']:.2f} "
                 f"ticks={condition_result['ma_ippo']['mean_ticks']:.1f}")
    else:
        LOG.warning(f"  [ma_ippo] SKIPPED — no model at {model_dir}")
        condition_result['ma_ippo'] = None

    return condition_result


# ============================================================================
# FULL SWEEP
# ============================================================================
def run_full_sweep(
    tasks: Optional[List[str]] = None,
    obs_noise_keys: Optional[List[str]] = None,
    fail_regimes: Optional[List[str]] = None,
    n_eval_episodes: int = 500,
    train_missing: bool = False,
    output_path: Optional[Path] = None,
    model_suffix: Optional[str] = None,
    train_rounds: int = TRAIN_ROUNDS,
    train_steps: int = TRAIN_STEPS,
) -> Dict[str, Any]:
    """Run the full comparison sweep.

    Args:
        tasks: Task names to sweep (default: all 7)
        obs_noise_keys: Obs noise levels (default: all 3)
        fail_regimes: Failure regimes (default: all 5)
        n_eval_episodes: Episodes per condition per policy
        train_missing: Train MA-IPPO when models not found

    Returns:
        Full result dictionary
    """
    if tasks is None:
        tasks = TASKS
    if obs_noise_keys is None:
        obs_noise_keys = list(OBS_NOISE_LEVELS.keys())
    if fail_regimes is None:
        fail_regimes = FAIL_REGIMES

    task_defs = load_task_definitions()
    total_conditions = len(tasks) * len(obs_noise_keys) * len(fail_regimes)
    sweep_start = datetime.datetime.now()

    LOG.info(f"{'='*70}")
    LOG.info(f"Comparison 3-policy sweep started at {sweep_start.isoformat()}")
    LOG.info(f"  Tasks:       {tasks}")
    LOG.info(f"  Obs noise:   {obs_noise_keys}")
    LOG.info(f"  Fail regimes:{fail_regimes}")
    LOG.info(f"  Episodes:    {n_eval_episodes}")
    LOG.info(f"  Train missing: {train_missing}")
    LOG.info(f"  Total conditions: {total_conditions}")
    LOG.info(f"{'='*70}")

    results: Dict[str, Any] = {
        'experiment': 'comparison_4policy',
        'started_at': sweep_start.isoformat(),
        'n_eval_episodes': n_eval_episodes,
        'train_missing': train_missing,
        'comm_regime': COMM_REGIME,
        'decay_regime': DECAY_REGIME,
        'tasks_requested': tasks,
        'obs_noise_keys': obs_noise_keys,
        'fail_regimes': fail_regimes,
        'conditions': {},
    }

    condition_idx = 0
    for task_name in tasks:
        if task_name not in task_defs:
            LOG.warning(f"Unknown task '{task_name}', skipping")
            continue
        task_def = task_defs[task_name]
        results['conditions'].setdefault(task_name, {})

        for obs_key in obs_noise_keys:
            results['conditions'][task_name].setdefault(obs_key, {})

            for fail_regime in fail_regimes:
                condition_idx += 1
                cond_start = time.time()
                LOG.info(f"\n[{condition_idx}/{total_conditions}] "
                         f"{task_name} / {obs_key} / {fail_regime}")

                condition_result = run_single_condition(
                    task_name=task_name,
                    task_def=task_def,
                    obs_noise_key=obs_key,
                    fail_regime=fail_regime,
                    n_eval_episodes=n_eval_episodes,
                    train_missing=train_missing,
                    model_suffix=model_suffix,
                    train_rounds=train_rounds,
                    train_steps=train_steps,
                )
                condition_result['condition_time_s'] = time.time() - cond_start
                results['conditions'][task_name][obs_key][fail_regime] = condition_result

                # Save incrementally
                _save_results(results, output_path)

    sweep_end = datetime.datetime.now()
    results['finished_at'] = sweep_end.isoformat()
    results['total_time_s'] = (sweep_end - sweep_start).total_seconds()
    _save_results(results, output_path)

    LOG.info(f"\n{'='*70}")
    LOG.info(f"Sweep finished at {sweep_end.isoformat()}")
    LOG.info(f"Total time: {results['total_time_s']:.1f}s")
    LOG.info(f"{'='*70}")

    return results


def _save_results(results: Dict[str, Any], output_path: Optional[Path] = None):
    """Save results to JSON file."""
    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "results" / "comparison_4policy.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Comparison experiment: Passive vs Heuristic vs MA-IPPO'
    )
    parser.add_argument(
        '--task', default=None,
        help='Single task to evaluate (default: all 7 tasks)'
    )
    parser.add_argument(
        '--obs-noise', default=None,
        choices=list(OBS_NOISE_LEVELS.keys()),
        help='Single observation noise level (default: all 3)'
    )
    parser.add_argument(
        '--obs-noise-list', default=None,
        help='Comma-separated obs noise keys (overrides --obs-noise)'
    )
    parser.add_argument(
        '--fail-regime', default=None,
        choices=FAIL_REGIMES,
        help='Single failure cost regime (default: all 5)'
    )
    parser.add_argument(
        '--fail-regimes', default=None,
        help='Comma-separated fail regimes (overrides --fail-regime)'
    )
    parser.add_argument(
        '--eval-episodes', type=int, default=500,
        help='Number of evaluation episodes per condition (default: 500)'
    )
    parser.add_argument(
        '--train-missing', action='store_true',
        help='Train MA-IPPO models when not found on disk'
    )
    parser.add_argument(
        '--train-rounds', type=int, default=TRAIN_ROUNDS,
        help=f'IPPO training rounds per condition (default: {TRAIN_ROUNDS})'
    )
    parser.add_argument(
        '--train-steps', type=int, default=TRAIN_STEPS,
        help=f'IPPO steps per round (default: {TRAIN_STEPS})'
    )
    parser.add_argument(
        '--model-suffix', default=None,
        help='Suffix for model root dir (e.g. "v3_semi_markov" → models/ma_ippo_v3_semi_markov/...)'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Run full sweep across all tasks, noise levels, and fail regimes'
    )
    parser.add_argument(
        '--output', default=None,
        help='Output JSON path (default: data/results/comparison_4policy.json)'
    )
    args = parser.parse_args()

    # Initialize logging
    global LOG
    log_dir = PROJECT_ROOT / "data" / "logs"
    LOG = setup_logging(log_dir)

    # Determine sweep scope (CSV lists take priority over single-value args)
    tasks = [args.task] if args.task else None
    if args.obs_noise_list:
        obs_keys = [s.strip() for s in args.obs_noise_list.split(',') if s.strip()]
        for k in obs_keys:
            if k not in OBS_NOISE_LEVELS:
                LOG.error(f"Unknown obs_noise '{k}'. Valid: {list(OBS_NOISE_LEVELS.keys())}")
                sys.exit(1)
    elif args.obs_noise:
        obs_keys = [args.obs_noise]
    else:
        obs_keys = None

    if args.fail_regimes:
        fail_regimes = [s.strip() for s in args.fail_regimes.split(',') if s.strip()]
        for r in fail_regimes:
            if r not in FAIL_REGIMES:
                LOG.error(f"Unknown fail_regime '{r}'. Valid: {FAIL_REGIMES}")
                sys.exit(1)
    elif args.fail_regime:
        fail_regimes = [args.fail_regime]
    else:
        fail_regimes = None

    # Validate task
    if args.task:
        all_tasks = load_task_definitions()
        if args.task not in all_tasks:
            LOG.error(f"Unknown task '{args.task}'. Available: {list(all_tasks.keys())}")
            sys.exit(1)

    output_path = Path(args.output) if args.output else None

    results = run_full_sweep(
        tasks=tasks,
        obs_noise_keys=obs_keys,
        fail_regimes=fail_regimes,
        n_eval_episodes=args.eval_episodes,
        train_missing=args.train_missing,
        output_path=output_path,
        model_suffix=args.model_suffix,
        train_rounds=args.train_rounds,
        train_steps=args.train_steps,
    )

    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "results" / "comparison_4policy.json"
    LOG.info(f"Results saved to {output_path}")

    # Summary
    n_total = 0
    n_with_ippo = 0
    for task_data in results['conditions'].values():
        for obs_data in task_data.values():
            for fail_data in obs_data.values():
                n_total += 1
                if fail_data.get('ma_ippo') is not None:
                    n_with_ippo += 1
    LOG.info(f"Evaluated {n_total} conditions ({n_with_ippo} with MA-IPPO)")


if __name__ == '__main__':
    main()
