"""
Multi-Agent IPPO Training for the Procedure Assistant Simulation.

Implements Independent PPO (IPPO) with alternating best-response training:
  - Round 1: Train human while assistant is fixed (random or from previous round)
  - Round 2: Train assistant while human is fixed
  - Repeat until convergence

Usage:
    cd src/training
    python train_ma_ippo.py --task make_cereal --regime balanced --rounds 10

Output:
    models/ma_ippo/<task>/<regime>/
        human_model_final.zip
        assistant_model_final.zip
        human_model_best.zip
        assistant_model_best.zip
    data/results/ma_ippo_<task>_<regime>.json
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from typing import Dict, Any, Optional, Tuple
import json
import argparse
import sys
from pathlib import Path

# Path setup
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))

from ma_procedure_assistant_sim import (
    MAProcedureAssistantEnv,
    MASimulationParams,
    RandomHumanPolicy,
    RandomAssistantPolicy,
    PassiveHumanPolicy,
    SilentAssistantPolicy,
)
from task_definitions import TaskDefinition, load_task_definitions


# ============================================================================
# GYM WRAPPERS
# ============================================================================
class HumanGymWrapper(gym.Env):
    """Gymnasium wrapper for the human agent.

    The human observes [current_step, tau, memory_current, assistant_last_action,
    obs_noise_state] as a flat float32 array (shape: 5).

    The assistant policy is held fixed during human training; it is called
    inside step() to generate the joint action.

    Args:
        ma_env: Shared MAProcedureAssistantEnv instance
        assistant_policy: Policy with .predict(obs) → (action, _) interface
    """

    def __init__(self, ma_env: MAProcedureAssistantEnv, assistant_policy):
        super().__init__()
        self.ma_env = ma_env
        self.assistant_policy = assistant_policy

        n_steps = ma_env.n_steps

        # v4 Observation (fixed 8 scalars): [current_identity, tau,
        #   memory_current, memory_next, asst_last_action, obs_noise,
        #   narrated_current_count, asked_next_count]
        max_count = 10.0  # per-step habituation counts saturate well before this
        obs_low = np.zeros(8, dtype=np.float32)
        obs_high = np.array(
            [n_steps, 200, 2.0, 2.0, ma_env.n_assistant_actions - 1, 1.0,
             max_count, max_count],
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # Action: {silent, narrate, question_next}
        self.action_space = gym.spaces.Discrete(ma_env.n_human_actions)

        self._last_assistant_obs: Optional[np.ndarray] = None
        self._episode_reward: float = 0.0

    def _convert_human_obs(self, obs_dict: Dict) -> np.ndarray:
        # v4: fixed 8 scalars (no per-step vectors on the human side)
        return np.array(
            [
                obs_dict['current_identity'],
                obs_dict['tau'],
                obs_dict['memory_current'],
                obs_dict['memory_next'],
                obs_dict['assistant_last_action'],
                obs_dict['obs_noise_state'],
                obs_dict['narrated_current_count'],
                obs_dict['asked_next_count'],
            ],
            dtype=np.float32,
        )

    def _convert_assistant_obs(self, obs_dict: Dict) -> np.ndarray:
        # v4: semi-Markov belief — step_belief(N+1) + expected_tau(N+1)
        #     + memory_estimate(N) + human_last_action(1)
        #     + asked_count(N) + reminded_count(N)
        return np.concatenate([
            obs_dict['step_belief'].astype(np.float32),
            obs_dict['expected_tau'].astype(np.float32),
            obs_dict['memory_estimate'].astype(np.float32),
            np.array([obs_dict['human_last_action']], dtype=np.float32),
            obs_dict['asked_count'].astype(np.float32),
            obs_dict['reminded_count'].astype(np.float32),
        ])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        h_obs_dict, a_obs_dict = self.ma_env.reset()
        self._last_assistant_obs = self._convert_assistant_obs(a_obs_dict)
        self._episode_reward = 0.0
        return self._convert_human_obs(h_obs_dict), {}

    def step(self, human_action: int):
        # Get assistant's action from the fixed policy
        a_action, _ = self.assistant_policy.predict(self._last_assistant_obs)
        a_action = int(a_action)

        h_obs_dict, a_obs_dict, reward, done, info = self.ma_env.step(
            human_action, a_action
        )
        self._last_assistant_obs = self._convert_assistant_obs(a_obs_dict)
        self._episode_reward += reward

        if done:
            info['episode'] = {'r': self._episode_reward, 'l': self.ma_env.ma_state.global_tick}

        return self._convert_human_obs(h_obs_dict), reward, done, False, info


class AssistantGymWrapper(gym.Env):
    """Gymnasium wrapper for the assistant agent.

    The assistant observes [step_estimate, elapsed_time, memory[0..N-1],
    obs_noise_state, human_last_action] as a flat float32 array (shape: N+4).

    The human policy is held fixed during assistant training.

    Args:
        ma_env: Shared MAProcedureAssistantEnv instance
        human_policy: Policy with .predict(obs) → (action, _) interface
    """

    def __init__(self, ma_env: MAProcedureAssistantEnv, human_policy):
        super().__init__()
        self.ma_env = ma_env
        self.human_policy = human_policy

        n_steps = ma_env.n_steps

        # v4 Observation: [step_belief(N+1), expected_tau(N+1),
        #   memory_estimate(N), human_last_action(1), asked_count(N),
        #   reminded_count(N)]
        # shape: (2(N+1) + 3N + 1,)
        tau_max = ma_env.tau_max
        # Upper bound on per-step prior counts: 10. Costs at growth=2.0,
        # cap=27 saturate the cost by ~4 anyway, so 10 is a safe obs ceiling.
        max_count = 10.0
        obs_low = np.zeros(2 * (n_steps + 1) + 3 * n_steps + 1, dtype=np.float32)
        obs_high = np.array(
            [1.0] * (n_steps + 1)                  # step_belief: probabilities
            + [float(tau_max)] * (n_steps + 1)      # expected_tau: 0..tau_max
            + [2.0] * n_steps                       # memory_estimate (all steps)
            + [ma_env.n_human_actions - 1]          # human_last_action
            + [max_count] * n_steps                 # asked_count per step
            + [max_count] * n_steps,                # reminded_count per step
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # Action: {silent, confirm, remind_0, ..., remind_{N-1}}  (v4: all steps)
        self.action_space = gym.spaces.Discrete(ma_env.n_assistant_actions)

        self._last_human_obs: Optional[np.ndarray] = None
        self._episode_reward: float = 0.0

    def _convert_human_obs(self, obs_dict: Dict) -> np.ndarray:
        # v4: fixed 8 scalars (no per-step vectors on the human side)
        return np.array(
            [
                obs_dict['current_identity'],
                obs_dict['tau'],
                obs_dict['memory_current'],
                obs_dict['memory_next'],
                obs_dict['assistant_last_action'],
                obs_dict['obs_noise_state'],
                obs_dict['narrated_current_count'],
                obs_dict['asked_next_count'],
            ],
            dtype=np.float32,
        )

    def _convert_assistant_obs(self, obs_dict: Dict) -> np.ndarray:
        # v4: semi-Markov belief — step_belief(N+1) + expected_tau(N+1)
        #     + memory_estimate(N) + human_last_action(1)
        #     + asked_count(N) + reminded_count(N)
        return np.concatenate([
            obs_dict['step_belief'].astype(np.float32),
            obs_dict['expected_tau'].astype(np.float32),
            obs_dict['memory_estimate'].astype(np.float32),
            np.array([obs_dict['human_last_action']], dtype=np.float32),
            obs_dict['asked_count'].astype(np.float32),
            obs_dict['reminded_count'].astype(np.float32),
        ])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        h_obs_dict, a_obs_dict = self.ma_env.reset()
        self._last_human_obs = self._convert_human_obs(h_obs_dict)
        self._episode_reward = 0.0
        return self._convert_assistant_obs(a_obs_dict), {}

    def step(self, assistant_action: int):
        # Get human's action from the fixed policy
        h_action, _ = self.human_policy.predict(self._last_human_obs)
        h_action = int(h_action)

        h_obs_dict, a_obs_dict, reward, done, info = self.ma_env.step(
            h_action, assistant_action
        )
        self._last_human_obs = self._convert_human_obs(h_obs_dict)
        self._episode_reward += reward

        if done:
            info['episode'] = {'r': self._episode_reward, 'l': self.ma_env.ma_state.global_tick}

        return self._convert_assistant_obs(a_obs_dict), reward, done, False, info


# ============================================================================
# JOINT EVALUATION
# ============================================================================
def evaluate_joint(
    human_model,
    assistant_model,
    ma_env: MAProcedureAssistantEnv,
    n_episodes: int = 100,
) -> Dict[str, float]:
    """Evaluate the joint policy over multiple episodes.

    Args:
        human_model: Trained human PPO model (or policy with .predict())
        assistant_model: Trained assistant PPO model (or policy with .predict())
        ma_env: Shared multi-agent environment
        n_episodes: Number of evaluation episodes

    Returns:
        Dictionary with metrics: mean_reward, mean_failures, mean_narrations,
        mean_questions, mean_interactions, mean_ticks
    """
    human_wrapper = HumanGymWrapper(ma_env, assistant_model)
    assistant_wrapper = AssistantGymWrapper(ma_env, human_model)

    total_rewards = []
    total_failures = []
    total_narrations = []
    total_questions = []
    total_interactions = []
    total_reminds = []
    total_confirms = []
    total_ticks = []
    total_tracking_acc = []

    for ep in range(n_episodes):
        h_obs_dict, a_obs_dict = ma_env.reset()

        h_obs = human_wrapper._convert_human_obs(h_obs_dict)
        a_obs = assistant_wrapper._convert_assistant_obs(a_obs_dict)

        ep_reward = 0.0
        ep_tracking = []
        done = False

        while not done:
            h_action, _ = human_model.predict(h_obs, deterministic=True)
            a_action, _ = assistant_model.predict(a_obs, deterministic=True)

            h_obs_dict, a_obs_dict, reward, done, info = ma_env.step(
                int(h_action), int(a_action)
            )

            h_obs = human_wrapper._convert_human_obs(h_obs_dict)
            a_obs = assistant_wrapper._convert_assistant_obs(a_obs_dict)
            ep_reward += reward
            mc = info.get('tracking_map_correct')
            if mc is not None:
                ep_tracking.append(mc)

        state = ma_env.ma_state
        total_rewards.append(ep_reward)
        total_failures.append(state.total_failures)
        total_narrations.append(state.total_narrations)
        total_questions.append(state.total_questions)
        total_interactions.append(state.total_interactions)
        total_reminds.append(state.total_reminds)
        total_confirms.append(state.total_confirms)
        total_ticks.append(state.global_tick)
        total_tracking_acc.append(float(np.mean(ep_tracking)) if ep_tracking else np.nan)

    return {
        'mean_reward': float(np.mean(total_rewards)),
        'std_reward': float(np.std(total_rewards)),
        'mean_failures': float(np.mean(total_failures)),
        'mean_narrations': float(np.mean(total_narrations)),
        'mean_questions': float(np.mean(total_questions)),
        'mean_interactions': float(np.mean(total_interactions)),
        'mean_reminds': float(np.mean(total_reminds)),
        'mean_confirms': float(np.mean(total_confirms)),
        'mean_ticks': float(np.mean(total_ticks)),
        'mean_tracking_map_acc': float(np.nanmean(total_tracking_acc)),
        'std_tracking_map_acc': float(np.nanstd(total_tracking_acc)),
    }


# ============================================================================
# IPPO ALTERNATING TRAINING
# ============================================================================
def train_ippo(
    params: MASimulationParams,
    task_def: TaskDefinition,
    n_rounds: int = 30,
    steps_per_round: int = 50_000,
    save_dir: Optional[Path] = None,
    verbose: int = 1,
    min_rounds: int = 8,
    patience: int = 6,
    min_delta: float = 0.5,
    eval_episodes: int = 100,
) -> Tuple[PPO, PPO, Dict[str, Any]]:
    """Train both agents using alternating IPPO (best-response iteration)
    with early stopping and best-checkpoint return.

    Each round:
    1. Train human policy with assistant fixed
    2. Train assistant policy with human fixed
    3. Evaluate joint policy over `eval_episodes` episodes; track the best.

    Early stopping: after at least `min_rounds`, stop if the best joint-eval
    reward has not improved by > `min_delta` for `patience` consecutive rounds
    (handles both "still improving -> keep going" and "saturated/oscillating
    -> stop"). The returned models are the BEST checkpoint (reloaded from
    disk), not the final round — so downstream evaluation reports the best
    policy found, which matters for oscillating runs.

    Args:
        params: MASimulationParams
        task_def: Task to train on
        n_rounds: MAX alternating training rounds (upper bound)
        steps_per_round: Environment steps per agent per round
        save_dir: Directory to save models (best + final). Required for the
                  best-checkpoint return; if None, returns the final models.
        verbose: Verbosity level for PPO (0=quiet, 1=info)
        min_rounds: do not early-stop before this many rounds
        patience: stop after this many consecutive rounds without improvement
        min_delta: minimum reward gain to count as an improvement
        eval_episodes: episodes for the per-round joint evaluation

    Returns:
        (human_model, assistant_model, training_log)  — models are BEST checkpoint
    """
    ma_env = MAProcedureAssistantEnv(params, task_def)

    if verbose:
        print(f"Task: {task_def.task_name} ({task_def.n_steps} steps, "
              f"{ma_env.n_critical} critical)")
        print(f"Human actions: {ma_env.n_human_actions}  "
              f"Assistant actions: {ma_env.n_assistant_actions}")
        print(f"R_complete = {ma_env.R_complete:.1f}")

    # --- Initialize with random policies ---
    random_assistant = RandomAssistantPolicy(ma_env.n_assistant_actions)
    random_human = RandomHumanPolicy(ma_env.n_human_actions)

    human_env_init = Monitor(HumanGymWrapper(ma_env, random_assistant))
    assistant_env_init = Monitor(AssistantGymWrapper(ma_env, random_human))

    human_model = PPO(
        "MlpPolicy",
        human_env_init,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )
    assistant_model = PPO(
        "MlpPolicy",
        assistant_env_init,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    training_log: Dict[str, Any] = {
        'task': task_def.task_name,
        'n_rounds': n_rounds,
        'steps_per_round': steps_per_round,
        'params': params.to_dict(),
        'rounds': [],
    }

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    best_reward = -np.inf
    best_human_path = None
    best_assistant_path = None
    best_round = None
    since_improve = 0
    stopped_round = n_rounds

    for round_idx in range(n_rounds):
        if verbose:
            print(f"\n--- Round {round_idx + 1}/{n_rounds} ---")

        # Train human (assistant fixed)
        human_env = Monitor(HumanGymWrapper(ma_env, assistant_model))
        human_model.set_env(human_env)
        human_model.learn(total_timesteps=steps_per_round, reset_num_timesteps=False)
        if verbose:
            print(f"  Human trained ({steps_per_round} steps)")

        # Train assistant (human fixed)
        assistant_env = Monitor(AssistantGymWrapper(ma_env, human_model))
        assistant_model.set_env(assistant_env)
        assistant_model.learn(total_timesteps=steps_per_round, reset_num_timesteps=False)
        if verbose:
            print(f"  Assistant trained ({steps_per_round} steps)")

        # Evaluate joint policy
        eval_metrics = evaluate_joint(human_model, assistant_model, ma_env,
                                      n_episodes=eval_episodes)
        training_log['rounds'].append({'round': round_idx + 1, **eval_metrics})

        if verbose:
            print(
                f"  Joint eval → reward={eval_metrics['mean_reward']:.2f}, "
                f"failures={eval_metrics['mean_failures']:.2f}, "
                f"narrations={eval_metrics['mean_narrations']:.2f}, "
                f"questions={eval_metrics['mean_questions']:.2f}"
            )

        # Track best (improvement = gain > min_delta) and save best checkpoint
        improved = eval_metrics['mean_reward'] > best_reward + min_delta
        if improved:
            best_reward = eval_metrics['mean_reward']
            best_round = round_idx + 1
            since_improve = 0
            if save_dir is not None:
                best_human_path = save_dir / "human_model_best"
                best_assistant_path = save_dir / "assistant_model_best"
                human_model.save(str(best_human_path))
                assistant_model.save(str(best_assistant_path))
                if verbose:
                    print(f"  New best (round {best_round}, "
                          f"reward={best_reward:.2f})! Saved to {save_dir}")
        else:
            since_improve += 1

        # Early stopping: only after min_rounds, stop if stalled for `patience`
        if (round_idx + 1) >= min_rounds and since_improve >= patience:
            stopped_round = round_idx + 1
            if verbose:
                print(f"  Early stop at round {stopped_round} "
                      f"(no improvement for {patience} rounds; "
                      f"best round {best_round}, reward {best_reward:.2f})")
            break

    training_log['best_round'] = best_round
    training_log['best_round_reward'] = float(best_reward)
    training_log['stopped_round'] = stopped_round

    # Save final models, then reload the BEST checkpoint to return it.
    if save_dir is not None:
        human_model.save(str(save_dir / "human_model_final"))
        assistant_model.save(str(save_dir / "assistant_model_final"))
        if verbose:
            print(f"\nFinal models saved to {save_dir}")
        if best_human_path is not None:
            human_model = PPO.load(str(best_human_path))
            assistant_model = PPO.load(str(best_assistant_path))
            if verbose:
                print(f"Returning BEST checkpoint (round {best_round}) for evaluation.")

    return human_model, assistant_model, training_log


# ============================================================================
# COST REGIMES (mirrors existing single-agent setup)
# ============================================================================
def define_ma_cost_regimes(
    task_def: TaskDefinition,
    comm_regime: str = 'default',
    decay_regime: str = 'default',
    obs_regime: str = 'default',
) -> Dict[str, MASimulationParams]:
    """Define failure cost regimes for multi-agent experiments.

    Builds params for all 3 failure cost regimes using the specified
    communication cost, memory decay, and observation noise regimes.

    Args:
        task_def: Task definition
        comm_regime: Key into COMM_COST_REGIMES
        decay_regime: Key into MEMORY_DECAY_REGIMES
        obs_regime: Key into OBS_NOISE_REGIMES

    Returns:
        Dict mapping failure regime name to MASimulationParams
    """
    from regime_definitions import build_params, FAILURE_COST_SCALES

    return {
        name: build_params(task_def, name, comm_regime, decay_regime, obs_regime)
        for name in FAILURE_COST_SCALES
    }


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Train MA-IPPO for procedure assistant')
    parser.add_argument('--task', default='make_cereal',
                        help='Task name (default: make_cereal)')
    parser.add_argument('--regime', default='balanced',
                        choices=['extremely_low', 'balanced', 'extremely_high'],
                        help='Cost regime (default: balanced)')
    parser.add_argument('--rounds', type=int, default=10,
                        help='Number of alternating training rounds (default: 10)')
    parser.add_argument('--steps', type=int, default=50_000,
                        help='Steps per agent per round (default: 50000)')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity (0=quiet, 1=info)')
    args = parser.parse_args()

    # Load task
    task_defs = load_task_definitions()
    if args.task not in task_defs:
        print(f"Unknown task '{args.task}'. Available: {list(task_defs.keys())}")
        sys.exit(1)
    task_def = task_defs[args.task]

    # Get params for regime
    regimes = define_ma_cost_regimes(task_def)
    if args.regime not in regimes:
        print(f"Unknown regime '{args.regime}'.")
        sys.exit(1)
    params = regimes[args.regime]

    # Output directories
    model_dir = PROJECT_ROOT / "models" / "ma_ippo" / args.task / args.regime
    results_dir = PROJECT_ROOT / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Train
    human_model, assistant_model, log = train_ippo(
        params=params,
        task_def=task_def,
        n_rounds=args.rounds,
        steps_per_round=args.steps,
        save_dir=model_dir,
        verbose=args.verbose,
    )

    # Save training log
    result_path = results_dir / f"ma_ippo_{args.task}_{args.regime}.json"
    with open(result_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining log saved to {result_path}")

    # Final evaluation
    ma_env = MAProcedureAssistantEnv(params, task_def)
    final_metrics = evaluate_joint(human_model, assistant_model, ma_env, n_episodes=200)
    print("\n=== Final Evaluation (200 episodes) ===")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.3f}")


if __name__ == '__main__':
    main()
