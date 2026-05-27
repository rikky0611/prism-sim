"""
Sensing-Fidelity Grid: obs_noise (baseline) × lambda_noise_recover (durability)

Sweeps the assistant's observation-noise dynamics at fixed communication and
failure cost. Tests claim C3: as narration becomes brittle (lambda_n large,
noise re-accumulates fast), the assistant must compensate with reminders;
durable narration shifts the load to the human.

Usage:
    # smoke test
    python run_sensing_grid.py --task make_cereal \
        --n-noise 2 --n-lambda 2 --rounds 2 --steps 5000 --eval-episodes 10

    # full sweep
    python run_sensing_grid.py --task make_cereal --n-noise 4 --n-lambda 4

Output:
    data/results/sensing_grid_<task>_cf<scale>_cc<comm>_seed<s>.json
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


def run_sensing_grid(
    task_name: str = 'make_cereal',
    n_noise: int = 4,
    n_lambda: int = 4,
    noise_vals: list = None,
    lambda_vals: list = None,
    obs_noise_min: float = 0.05,
    c_comm: float = 0.5,
    c_remind: float = 1.0,
    c_confirm: float = 1.0,
    c_off_timing: float = 3.0,
    c_fail_scale: float = 15.0,
    decay_regime: str = 'step_transition',
    n_rounds: int = 5,
    steps_per_round: int = 15_000,
    n_eval_episodes: int = 50,
    seed: int = 0,
    resume_path: str = None,
) -> dict:
    """Sweep (obs_noise, lambda_noise_recover) and train MA-IPPO at each cell."""
    tasks = load_task_definitions()
    task_def = tasks[task_name]

    if noise_vals is None:
        # Linear sweep across realistic noise levels
        noise_vals = list(np.linspace(0.2, 0.8, n_noise))
    if lambda_vals is None:
        # Log sweep: 0.02 (durable, ~35-tick half-life) → 0.20 (brittle, ~3.5-tick)
        lambda_vals = list(np.geomspace(0.02, 0.20, n_lambda))

    n_noise_actual = len(noise_vals)
    n_lambda_actual = len(lambda_vals)
    total = n_noise_actual * n_lambda_actual

    print(f"\n{'='*70}")
    print(f"Sensing grid: task={task_name}  {n_lambda_actual}×{n_noise_actual} = {total} pts")
    print(f"  obs_noise (baseline):     {noise_vals}")
    print(f"  lambda_noise_recover:     {[f'{v:.3f}' for v in lambda_vals]}")
    print(f"  fixed: c_comm={c_comm}  c_remind={c_remind}  c_confirm={c_confirm}")
    print(f"         c_fail_scale={c_fail_scale}  decay={decay_regime}  seed={seed}")
    print(f"  training: {n_rounds} rounds × {steps_per_round} steps  eval={n_eval_episodes}")
    print(f"{'='*70}\n")

    completed = {}
    if resume_path and Path(resume_path).exists():
        prev = json.load(open(resume_path))
        for row in prev.get('grid', []):
            for cell in row:
                if cell is not None:
                    key = (cell['i_lambda'], cell['i_noise'])
                    completed[key] = cell
        print(f"Resuming: {len(completed)} / {total} already done")

    np.random.seed(seed)

    grid = [[None] * n_noise_actual for _ in range(n_lambda_actual)]
    t0_total = time.time()
    done = 0

    for i_lambda, lam in enumerate(lambda_vals):
        for i_noise, noise in enumerate(noise_vals):
            key = (i_lambda, i_noise)
            if key in completed:
                grid[i_lambda][i_noise] = completed[key]
                done += 1
                continue

            t0 = time.time()
            params = build_params_sensing(
                task_def,
                c_fail_scale=c_fail_scale,
                c_comm=c_comm,
                obs_noise=float(noise),
                lambda_noise_recover=float(lam),
                obs_noise_min=obs_noise_min,
                c_remind=c_remind,
                c_confirm=c_confirm,
                c_off_timing=c_off_timing,
                decay_regime=decay_regime,
            )
            regime_name = f'sens_n{noise:.2f}_l{lam:.4f}_seed{seed}'
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
                'i_noise': i_noise,
                'obs_noise': float(noise),
                'lambda_noise_recover': float(lam),
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
            grid[i_lambda][i_noise] = cell

            human = cell['mean_narrations'] + cell['mean_questions']
            asst = cell['mean_reminds'] + cell['mean_confirms']
            total_comm = human + asst
            dol = (human / total_comm) if total_comm > 1e-6 else float('nan')
            eta = (time.time() - t0_total) / done * (total - done)
            print(f"[{done:3d}/{total}] n={noise:.2f} l={lam:.4f} | "
                  f"r={cell['mean_reward']:+.2f} "
                  f"nar={cell['mean_narrations']:.2f} "
                  f"rem={cell['mean_reminds']:.2f} "
                  f"con={cell['mean_confirms']:.2f} "
                  f"DoL={dol:.2f} "
                  f"trk={cell['mean_tracking_map_acc']:.3f} "
                  f"fail={cell['mean_failures']:.2f} | "
                  f"{elapsed:.1f}s  ETA {eta/60:.1f}min")

    return {
        'experiment': 'sensing_fidelity_grid',
        'task': task_name,
        'decay_regime': decay_regime,
        'c_fail_scale': c_fail_scale,
        'c_comm': c_comm,
        'c_remind': c_remind,
        'c_confirm': c_confirm,
        'c_off_timing': c_off_timing,
        'obs_noise_min': obs_noise_min,
        'seed': seed,
        'noise_vals': list(map(float, noise_vals)),
        'lambda_vals': list(map(float, lambda_vals)),
        'n_rounds': n_rounds,
        'steps_per_round': steps_per_round,
        'n_eval_episodes': n_eval_episodes,
        'grid': grid,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Sensing-fidelity grid: obs_noise × lambda_n'
    )
    parser.add_argument('--task', default='make_cereal')
    parser.add_argument('--n-noise', type=int, default=4)
    parser.add_argument('--n-lambda', type=int, default=4)
    parser.add_argument('--c-comm', type=float, default=0.5)
    parser.add_argument('--c-remind', type=float, default=1.0)
    parser.add_argument('--c-confirm', type=float, default=1.0)
    parser.add_argument('--c-off-timing', type=float, default=3.0)
    parser.add_argument('--c-fail-scale', type=float, default=15.0)
    parser.add_argument('--obs-noise-min', type=float, default=0.05)
    parser.add_argument('--decay-regime', default='step_transition')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--steps', type=int, default=15_000)
    parser.add_argument('--eval-episodes', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()

    results = run_sensing_grid(
        task_name=args.task,
        n_noise=args.n_noise,
        n_lambda=args.n_lambda,
        c_comm=args.c_comm,
        c_remind=args.c_remind,
        c_confirm=args.c_confirm,
        c_off_timing=args.c_off_timing,
        c_fail_scale=args.c_fail_scale,
        obs_noise_min=args.obs_noise_min,
        decay_regime=args.decay_regime,
        n_rounds=args.rounds,
        steps_per_round=args.steps,
        n_eval_episodes=args.eval_episodes,
        seed=args.seed,
        resume_path=args.resume,
    )

    out_dir = PROJECT_ROOT / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        f"sensing_grid_{args.task}_cf{int(args.c_fail_scale)}"
        f"_cc{args.c_comm:.2f}_seed{args.seed}.json"
    )
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
