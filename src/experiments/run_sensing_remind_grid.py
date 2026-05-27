"""
E3-v2: Brittleness × Reminder-cost grid

Sweeps (lambda_noise_recover × c_remind) at fixed baseline noise to test the
role-swap hypothesis directly: as narration becomes brittle (lambda ↑) AND
reminders are affordable (c_remind ↓), the assistant should compensate with
reminders. Original E3 (obs_noise × lambda) held c_remind=1.0 and reminders
were dominated by silence; this grid lowers c_remind to expose the swap.

Fixed:
    obs_noise (baseline) = 0.4
    c_nar = 0.5
    c_fail_scale = 15.0
    decay = step_transition

Swept:
    lambda_n  ∈ {0.02, 0.05, 0.10, 0.20}    (durable → brittle)
    c_remind  ∈ {0.05, 0.20, 0.50, 1.00}    (cheap → expensive)

Usage:
    python run_sensing_remind_grid.py --task make_cereal \
        --n-lambda 4 --n-c-remind 4 --rounds 4 --steps 10000 --eval-episodes 30 --seed 0
"""

import sys
import json
import argparse
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training"))

from task_definitions import load_task_definitions
from regime_definitions import build_params_sensing
from run_ma_experiments import run_experiment_for_regime


def run_sensing_remind_grid(
    task_name: str = 'make_cereal',
    n_lambda: int = 4,
    n_c_remind: int = 4,
    lambda_min: float = 0.02,
    lambda_max: float = 0.20,
    c_remind_min: float = 0.05,
    c_remind_max: float = 1.0,
    obs_noise: float = 0.4,
    obs_noise_min: float = 0.05,
    c_nar: float = 0.5,
    c_off_timing: float = 3.0,
    c_fail_scale: float = 15.0,
    decay_regime: str = 'step_transition',
    n_rounds: int = 4,
    steps_per_round: int = 10_000,
    n_eval_episodes: int = 30,
    seed: int = 0,
    resume_path: str = None,
) -> dict:
    tasks = load_task_definitions()
    task_def = tasks[task_name]

    lambda_vals = np.geomspace(lambda_min, lambda_max, n_lambda).tolist()
    c_remind_vals = np.geomspace(c_remind_min, c_remind_max, n_c_remind).tolist()
    total = n_lambda * n_c_remind

    print(f"\n{'='*70}")
    print(f"E3-v2 sensing×reminder grid: task={task_name}  "
          f"{n_lambda}×{n_c_remind} = {total} pts")
    print(f"  lambda_n  (durable→brittle):  "
          f"{[f'{v:.3f}' for v in lambda_vals]}")
    print(f"  c_remind  (cheap→expensive):  "
          f"{[f'{v:.3f}' for v in c_remind_vals]}")
    print(f"  fixed: obs_noise={obs_noise}  c_nar={c_nar}  "
          f"c_fail_scale={c_fail_scale}")
    print(f"  decay={decay_regime}  seed={seed}")
    print(f"  training: {n_rounds} rounds × {steps_per_round} steps  "
          f"eval={n_eval_episodes}")
    print(f"{'='*70}\n")

    completed = {}
    if resume_path and Path(resume_path).exists():
        prev = json.load(open(resume_path))
        for row in prev.get('grid', []):
            for cell in row:
                if cell is not None:
                    key = (cell['i_lambda'], cell['i_c_remind'])
                    completed[key] = cell
        print(f"Resuming: {len(completed)} / {total} already done")

    np.random.seed(seed)

    grid = [[None] * n_c_remind for _ in range(n_lambda)]
    t0_total = time.time()
    done = 0

    for i_lambda, lam in enumerate(lambda_vals):
        for i_c_remind, cr in enumerate(c_remind_vals):
            key = (i_lambda, i_c_remind)
            if key in completed:
                grid[i_lambda][i_c_remind] = completed[key]
                done += 1
                continue

            t0 = time.time()
            params = build_params_sensing(
                task_def,
                c_fail_scale=c_fail_scale,
                c_comm=c_nar,
                obs_noise=float(obs_noise),
                lambda_noise_recover=float(lam),
                obs_noise_min=obs_noise_min,
                c_remind=float(cr),
                c_confirm=float(cr),
                c_off_timing=c_off_timing,
                decay_regime=decay_regime,
            )
            regime_name = f'sensrem_l{lam:.4f}_cr{cr:.4f}_seed{seed}'
            result = run_experiment_for_regime(
                task_name=task_name,
                regime_name=regime_name,
                params=params,
                task_def=task_def,
                n_rounds=n_rounds,
                steps_per_round=steps_per_round,
                n_eval_episodes=n_eval_episodes,
            )
            fe = result.get('ma_ippo', {}).get('final_eval', {})
            baseline = result.get('baselines', {}).get('both_silent', {})
            elapsed = time.time() - t0
            done += 1

            cell = {
                'i_lambda': i_lambda,
                'i_c_remind': i_c_remind,
                'lambda_noise_recover': float(lam),
                'c_remind': float(cr),
                'mean_reward':     fe.get('mean_reward', 0.0),
                'mean_narrations': fe.get('mean_narrations', 0.0),
                'mean_questions':  fe.get('mean_questions', 0.0),
                'mean_reminds':    fe.get('mean_reminds', 0.0),
                'mean_confirms':   fe.get('mean_confirms', 0.0),
                'mean_failures':   fe.get('mean_failures', 0.0),
                'mean_tracking_map_acc': fe.get('mean_tracking_map_acc', float('nan')),
                'baseline_mean_reward':   baseline.get('mean_reward'),
                'baseline_mean_failures': baseline.get('mean_failures'),
            }
            grid[i_lambda][i_c_remind] = cell

            human = cell['mean_narrations'] + cell['mean_questions']
            asst = cell['mean_reminds'] + cell['mean_confirms']
            total_comm = human + asst
            dol = (human / total_comm) if total_comm > 1e-6 else float('nan')
            eta = (time.time() - t0_total) / max(1, done) * (total - done)
            print(f"[{done:3d}/{total}] l={lam:.4f} cr={cr:.3f} | "
                  f"r={cell['mean_reward']:+.2f} "
                  f"nar={cell['mean_narrations']:.2f} "
                  f"rem={cell['mean_reminds']:.2f} "
                  f"con={cell['mean_confirms']:.2f} "
                  f"DoL={dol:.2f} "
                  f"trk={cell['mean_tracking_map_acc']:.3f} "
                  f"fail={cell['mean_failures']:.2f} | "
                  f"{elapsed:.1f}s  ETA {eta/60:.1f}min")

            # Periodic checkpoint
            if done % 4 == 0:
                _save({
                    'experiment': 'sensing_remind_grid',
                    'task': task_name,
                    'decay_regime': decay_regime,
                    'c_fail_scale': c_fail_scale,
                    'c_nar': c_nar,
                    'obs_noise': obs_noise,
                    'obs_noise_min': obs_noise_min,
                    'c_off_timing': c_off_timing,
                    'seed': seed,
                    'lambda_vals': list(map(float, lambda_vals)),
                    'c_remind_vals': list(map(float, c_remind_vals)),
                    'n_rounds': n_rounds,
                    'steps_per_round': steps_per_round,
                    'n_eval_episodes': n_eval_episodes,
                    'grid': grid,
                }, task_name, c_fail_scale, obs_noise, seed)

    return {
        'experiment': 'sensing_remind_grid',
        'task': task_name,
        'decay_regime': decay_regime,
        'c_fail_scale': c_fail_scale,
        'c_nar': c_nar,
        'obs_noise': obs_noise,
        'obs_noise_min': obs_noise_min,
        'c_off_timing': c_off_timing,
        'seed': seed,
        'lambda_vals': list(map(float, lambda_vals)),
        'c_remind_vals': list(map(float, c_remind_vals)),
        'n_rounds': n_rounds,
        'steps_per_round': steps_per_round,
        'n_eval_episodes': n_eval_episodes,
        'grid': grid,
    }


def _save(results: dict, task_name: str, c_fail_scale: float, obs_noise: float,
          seed: int) -> Path:
    out_dir = PROJECT_ROOT / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        f"sensing_remind_grid_{task_name}_cf{int(c_fail_scale)}"
        f"_n{obs_noise:.2f}_seed{seed}.json"
    )
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', default='make_cereal')
    p.add_argument('--n-lambda', type=int, default=4)
    p.add_argument('--n-c-remind', type=int, default=4)
    p.add_argument('--obs-noise', type=float, default=0.4)
    p.add_argument('--obs-noise-min', type=float, default=0.05)
    p.add_argument('--c-nar', type=float, default=0.5)
    p.add_argument('--c-off-timing', type=float, default=3.0)
    p.add_argument('--c-fail-scale', type=float, default=15.0)
    p.add_argument('--decay-regime', default='step_transition')
    p.add_argument('--rounds', type=int, default=4)
    p.add_argument('--steps', type=int, default=10_000)
    p.add_argument('--eval-episodes', type=int, default=30)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--resume', default=None)
    args = p.parse_args()

    results = run_sensing_remind_grid(
        task_name=args.task,
        n_lambda=args.n_lambda,
        n_c_remind=args.n_c_remind,
        obs_noise=args.obs_noise,
        obs_noise_min=args.obs_noise_min,
        c_nar=args.c_nar,
        c_off_timing=args.c_off_timing,
        c_fail_scale=args.c_fail_scale,
        decay_regime=args.decay_regime,
        n_rounds=args.rounds,
        steps_per_round=args.steps,
        n_eval_episodes=args.eval_episodes,
        seed=args.seed,
        resume_path=args.resume,
    )

    path = _save(results, args.task, args.c_fail_scale, args.obs_noise, args.seed)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
