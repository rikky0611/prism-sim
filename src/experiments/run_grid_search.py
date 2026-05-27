"""
2D Grid Search: c_comm (=c_nar=c_q) × c_fail_scale

Sweeps two axes with c_remind=1.0 fixed, trains MA-IPPO at each grid point,
and saves results for phase-diagram visualization.

Usage:
    # smoke test (2×2, 2 rounds)
    python run_grid_search.py --task make_cereal --n-comm 2 --n-fail 2 \
        --rounds 2 --steps 5000 --eval-episodes 10

    # full grid (overnight)
    python run_grid_search.py --task make_cereal

Output:
    data/results/grid_search_<task>_<decay>_<obs>.json
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
from regime_definitions import build_params_grid
from run_ma_experiments import run_experiment_for_regime


def run_grid_search(
    task_name: str = 'make_cereal',
    n_comm: int = 10,
    n_fail: int = 10,
    c_comm_min: float = 0.05,
    c_comm_max: float = 5.0,
    c_fail_min: float = 1.0,
    c_fail_max: float = 100.0,
    c_remind: float = 1.0,
    c_confirm: float = 1.0,
    c_off_timing: float = 3.0,
    decay_regime: str = 'step_transition',
    obs_regime: str = 'durable',
    n_rounds: int = 5,
    steps_per_round: int = 15_000,
    n_eval_episodes: int = 50,
    resume_path: str = None,
) -> dict:
    """Run full grid search and return results dict."""
    tasks = load_task_definitions()
    task_def = tasks[task_name]

    c_comm_vals = np.geomspace(c_comm_min, c_comm_max, n_comm).tolist()
    c_fail_vals = np.geomspace(c_fail_min, c_fail_max, n_fail).tolist()
    total = n_comm * n_fail

    print(f"\n{'='*70}")
    print(f"Grid search: task={task_name}  {n_fail}×{n_comm} = {total} points")
    print(f"  c_comm: {c_comm_min:.3f} → {c_comm_max:.1f}  ({n_comm} pts, log)")
    print(f"  c_fail: {c_fail_min:.1f} → {c_fail_max:.1f}  ({n_fail} pts, log)")
    print(f"  fixed: c_remind={c_remind}  c_confirm={c_confirm}  "
          f"c_off_timing={c_off_timing}")
    print(f"  decay={decay_regime}  obs={obs_regime}")
    print(f"  training: {n_rounds} rounds × {steps_per_round} steps  "
          f"eval={n_eval_episodes} eps")
    print(f"{'='*70}\n")

    # Load partial results if resuming
    completed = {}  # key: (i_fail, j_comm) → result dict
    if resume_path and Path(resume_path).exists():
        prev = json.load(open(resume_path))
        for row in prev.get('grid', []):
            for cell in row:
                if cell is not None:
                    key = (cell['i_fail'], cell['i_comm'])
                    completed[key] = cell
        print(f"Resuming: {len(completed)} / {total} already done")

    grid = [[None] * n_comm for _ in range(n_fail)]
    t0_total = time.time()
    done = 0

    for i_fail, c_fail in enumerate(c_fail_vals):
        for j_comm, c_comm in enumerate(c_comm_vals):
            key = (i_fail, j_comm)
            if key in completed:
                grid[i_fail][j_comm] = completed[key]
                done += 1
                continue

            t0 = time.time()
            params = build_params_grid(
                task_def, c_fail, c_comm,
                c_remind=c_remind, c_confirm=c_confirm,
                c_off_timing=c_off_timing,
                decay_regime=decay_regime, obs_regime=obs_regime,
            )
            regime_name = f'cf{c_fail:.2f}_cc{c_comm:.4f}'
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
                'i_fail': i_fail,
                'i_comm': j_comm,
                'c_fail_scale': c_fail,
                'c_comm': c_comm,
                'mean_reward':       fe.get('mean_reward', 0.0),
                'mean_narrations':   fe.get('mean_narrations', 0.0),
                'mean_questions':    fe.get('mean_questions', 0.0),
                'mean_interactions': fe.get('mean_interactions', 0.0),
                'mean_failures':     fe.get('mean_failures', 0.0),
                'baseline_mean_reward':   baseline.get('mean_reward'),
                'baseline_mean_failures': baseline.get('mean_failures'),
            }
            grid[i_fail][j_comm] = cell

            br = cell['baseline_mean_reward']
            diff_str = (f"Δr={cell['mean_reward'] - br:+.2f} "
                        if br is not None else "")
            eta = (time.time() - t0_total) / done * (total - done)
            print(f"[{done:3d}/{total}] cf={c_fail:6.1f} cc={c_comm:.4f} | "
                  f"r={cell['mean_reward']:+.2f} "
                  f"{diff_str}"
                  f"narr={cell['mean_narrations']:.2f} "
                  f"q={cell['mean_questions']:.2f} "
                  f"int={cell['mean_interactions']:.2f} "
                  f"fail={cell['mean_failures']:.2f} | "
                  f"{elapsed:.1f}s  ETA {eta/60:.1f}min")

    return {
        'task': task_name,
        'decay_regime': decay_regime,
        'obs_regime': obs_regime,
        'c_remind': c_remind,
        'c_confirm': c_confirm,
        'c_off_timing': c_off_timing,
        'c_comm_vals': c_comm_vals,
        'c_fail_vals': c_fail_vals,
        'n_rounds': n_rounds,
        'steps_per_round': steps_per_round,
        'n_eval_episodes': n_eval_episodes,
        'grid': grid,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Grid search: c_comm × c_fail_scale phase diagram'
    )
    parser.add_argument('--task', default='make_cereal')
    parser.add_argument('--n-comm', type=int, default=10,
                        help='Grid points along c_comm axis (default: 10)')
    parser.add_argument('--n-fail', type=int, default=10,
                        help='Grid points along c_fail axis (default: 10)')
    parser.add_argument('--c-comm-min', type=float, default=0.05)
    parser.add_argument('--c-comm-max', type=float, default=5.0)
    parser.add_argument('--c-fail-min', type=float, default=1.0)
    parser.add_argument('--c-fail-max', type=float, default=100.0)
    parser.add_argument('--c-remind', type=float, default=1.0)
    parser.add_argument('--c-confirm', type=float, default=1.0)
    parser.add_argument('--decay-regime', default='step_transition')
    parser.add_argument('--obs-regime', default='durable')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--steps', type=int, default=15_000)
    parser.add_argument('--eval-episodes', type=int, default=50)
    parser.add_argument('--resume', default=None,
                        help='Path to partial results JSON to resume from')
    args = parser.parse_args()

    results = run_grid_search(
        task_name=args.task,
        n_comm=args.n_comm,
        n_fail=args.n_fail,
        c_comm_min=args.c_comm_min,
        c_comm_max=args.c_comm_max,
        c_fail_min=args.c_fail_min,
        c_fail_max=args.c_fail_max,
        c_remind=args.c_remind,
        c_confirm=args.c_confirm,
        decay_regime=args.decay_regime,
        obs_regime=args.obs_regime,
        n_rounds=args.rounds,
        steps_per_round=args.steps,
        n_eval_episodes=args.eval_episodes,
        resume_path=args.resume,
    )

    out_dir = PROJECT_ROOT / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        f"grid_search_{args.task}_{args.decay_regime}_{args.obs_regime}.json"
    )
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
