"""
Cost-Asymmetry Grid: c_nar (human) × c_remind (assistant)

Sweeps the two AGENT-INDEPENDENT communication-cost axes, holding c_fail,
decay, and sensing fixed. Each cell trains MA-IPPO and records who pays
what — exposing the optimal *division of communicative labor*.

This is the headline figure for E1 (Communication Regime axis).

Usage:
    # smoke test (2x2, tiny training)
    python run_grid_asymmetric.py --task make_cereal \
        --n-c-nar 2 --n-c-remind 2 \
        --rounds 2 --steps 5000 --eval-episodes 10

    # full grid (overnight)
    python run_grid_asymmetric.py --task make_cereal --n-c-nar 8 --n-c-remind 8

Output:
    data/results/grid_asymmetric_<task>_<decay>_<obs>_cf<scale>_seed<s>.json
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
from regime_definitions import build_params_asymmetric
from run_ma_experiments import run_experiment_for_regime


def _build_results(task_name, decay_regime, obs_regime, c_fail_scale,
                   c_off_timing, seed, c_nar_vals, c_remind_vals,
                   n_rounds, steps_per_round, n_eval_episodes, grid):
    return {
        'experiment': 'cost_asymmetry_grid',
        'task': task_name,
        'decay_regime': decay_regime,
        'obs_regime': obs_regime,
        # v5 actual failure model: critical step failure is terminal with a
        # large fixed penalty; non-critical failure is soft and continues.
        # The legacy `c_fail_scale` field below is vestigial (kept for
        # backward compatibility / filename tags) and does NOT affect dynamics.
        'c_fail_critical': 50.0,
        'c_fail_noncritical': 1.0,
        'terminate_on_critical_failure': True,
        'c_fail_scale': c_fail_scale,  # legacy: vestigial in v5, no effect
        'c_off_timing': c_off_timing,
        'seed': seed,
        'c_nar_vals': c_nar_vals,
        'c_remind_vals': c_remind_vals,
        'n_rounds': n_rounds,
        'steps_per_round': steps_per_round,
        'n_eval_episodes': n_eval_episodes,
        'grid': grid,
    }


def run_grid_asymmetric(
    task_name: str = 'make_cereal',
    n_c_nar: int = 8,
    n_c_remind: int = 8,
    # Cost range calibrated to external validity anchor: 1 narration ~ 0.5-5s
    # of speaking time. Earlier default 0.05 corresponds to ~0.05s of speech,
    # which is below the realistic per-utterance floor and produced saturated
    # comm counts (~70+/ep) in the cheapest cells.
    c_nar_min: float = 0.5,
    c_nar_max: float = 5.0,
    c_remind_min: float = 0.5,
    c_remind_max: float = 5.0,
    c_fail_scale: float = 15.0,
    c_off_timing: float = 3.0,
    decay_regime: str = 'step_transition',
    obs_regime: str = 'durable',
    n_rounds: int = 30,
    steps_per_round: int = 15_000,
    n_eval_episodes: int = 300,
    seed: int = 0,
    resume_path: str = None,
    incremental_save_path: str = None,
    min_rounds: int = 8,
    patience: int = 6,
    eval_episodes: int = 100,
) -> dict:
    """Run the cost-asymmetry grid and return results dict."""
    tasks = load_task_definitions()
    task_def = tasks[task_name]

    c_nar_vals = np.geomspace(c_nar_min, c_nar_max, n_c_nar).tolist()
    c_remind_vals = np.geomspace(c_remind_min, c_remind_max, n_c_remind).tolist()
    total = n_c_nar * n_c_remind

    print(f"\n{'='*70}")
    print(f"Asymmetry grid: task={task_name}  {n_c_remind}×{n_c_nar} = {total} pts")
    print(f"  c_nar    (human):     {c_nar_min:.3f} → {c_nar_max:.1f}  ({n_c_nar} pts, log)")
    print(f"  c_remind (assistant): {c_remind_min:.3f} → {c_remind_max:.1f}  ({n_c_remind} pts, log)")
    print(f"  fixed: c_fail_scale={c_fail_scale}  c_off={c_off_timing}")
    print(f"  decay={decay_regime}  obs={obs_regime}  seed={seed}")
    print(f"  training: {n_rounds} rounds × {steps_per_round} steps  "
          f"eval={n_eval_episodes} eps")
    print(f"{'='*70}\n")

    completed = {}
    if resume_path and Path(resume_path).exists():
        prev = json.load(open(resume_path))
        for row in prev.get('grid', []):
            for cell in row:
                if cell is not None:
                    key = (cell['i_remind'], cell['i_nar'])
                    completed[key] = cell
        print(f"Resuming: {len(completed)} / {total} already done")

    # Optional seeding for reproducibility (PPO ignores numpy seed by default)
    np.random.seed(seed)

    grid = [[None] * n_c_nar for _ in range(n_c_remind)]
    t0_total = time.time()
    done = 0

    for i_remind, c_remind in enumerate(c_remind_vals):
        for i_nar, c_nar in enumerate(c_nar_vals):
            key = (i_remind, i_nar)
            if key in completed:
                grid[i_remind][i_nar] = completed[key]
                done += 1
                continue

            t0 = time.time()
            params = build_params_asymmetric(
                task_def,
                c_fail_scale=c_fail_scale,
                c_nar=c_nar,
                c_remind=c_remind,
                c_off_timing=c_off_timing,
                decay_regime=decay_regime,
                obs_regime=obs_regime,
            )
            regime_name = f'asym_cn{c_nar:.4f}_cr{c_remind:.4f}_seed{seed}'
            result = run_experiment_for_regime(
                task_name=task_name,
                regime_name=regime_name,
                params=params,
                task_def=task_def,
                n_rounds=n_rounds,
                steps_per_round=steps_per_round,
                n_eval_episodes=n_eval_episodes,
                min_rounds=min_rounds,
                patience=patience,
                eval_episodes=eval_episodes,
            )
            fe = result.get('ma_ippo', {}).get('final_eval', {})
            baseline = result.get('baselines', {}).get('both_silent', {})
            training_log = result.get('ma_ippo', {}).get('training_log', [])
            elapsed = time.time() - t0
            done += 1

            cell = {
                'i_remind': i_remind,
                'i_nar': i_nar,
                'c_nar': c_nar,
                'c_remind': c_remind,
                'mean_reward':       fe.get('mean_reward', 0.0),
                'mean_narrations':   fe.get('mean_narrations', 0.0),
                'mean_questions':    fe.get('mean_questions', 0.0),
                'mean_reminds':      fe.get('mean_reminds', 0.0),
                'mean_confirms':     fe.get('mean_confirms', 0.0),
                'mean_interactions': fe.get('mean_interactions', 0.0),
                'mean_failures':     fe.get('mean_failures', 0.0),
                'baseline_mean_reward':   baseline.get('mean_reward'),
                'baseline_mean_failures': baseline.get('mean_failures'),
                'training_log':           training_log,
                'best_round':             result.get('ma_ippo', {}).get('best_round'),
                'stopped_round':          result.get('ma_ippo', {}).get('stopped_round'),
            }
            grid[i_remind][i_nar] = cell

            # Division-of-labor index: human share of total comm
            human = cell['mean_narrations'] + cell['mean_questions']
            asst = cell['mean_reminds'] + cell['mean_confirms']
            total_comm = human + asst
            dol = (human / total_comm) if total_comm > 1e-6 else float('nan')
            eta = (time.time() - t0_total) / done * (total - done)
            print(f"[{done:3d}/{total}] cn={c_nar:.4f} cr={c_remind:.4f} | "
                  f"r={cell['mean_reward']:+.2f} "
                  f"nar={cell['mean_narrations']:.2f} "
                  f"q={cell['mean_questions']:.2f} "
                  f"rem={cell['mean_reminds']:.2f} "
                  f"con={cell['mean_confirms']:.2f} "
                  f"DoL={dol:.2f} "
                  f"fail={cell['mean_failures']:.2f} | "
                  f"{elapsed:.1f}s  ETA {eta/60:.1f}min")

            if incremental_save_path:
                partial = _build_results(
                    task_name, decay_regime, obs_regime, c_fail_scale,
                    c_off_timing, seed, c_nar_vals, c_remind_vals,
                    n_rounds, steps_per_round, n_eval_episodes, grid,
                )
                Path(incremental_save_path).parent.mkdir(parents=True, exist_ok=True)
                with open(incremental_save_path, 'w') as f:
                    json.dump(partial, f, indent=2)

    return _build_results(
        task_name, decay_regime, obs_regime, c_fail_scale,
        c_off_timing, seed, c_nar_vals, c_remind_vals,
        n_rounds, steps_per_round, n_eval_episodes, grid,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Cost-asymmetry grid: c_nar × c_remind phase diagram'
    )
    parser.add_argument('--task', default='make_cereal')
    parser.add_argument('--n-c-nar', type=int, default=8)
    parser.add_argument('--n-c-remind', type=int, default=8)
    parser.add_argument('--c-nar-min', type=float, default=0.5)
    parser.add_argument('--c-nar-max', type=float, default=5.0)
    parser.add_argument('--c-remind-min', type=float, default=0.5)
    parser.add_argument('--c-remind-max', type=float, default=5.0)
    parser.add_argument('--c-fail-scale', type=float, default=15.0)
    parser.add_argument('--c-off-timing', type=float, default=3.0)
    parser.add_argument('--decay-regime', default='step_transition')
    parser.add_argument('--obs-regime', default='durable')
    parser.add_argument('--rounds', type=int, default=30,
                        help='MAX training rounds (early stopping may stop sooner)')
    parser.add_argument('--steps', type=int, default=15_000)
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Per-round eval episodes (best-checkpoint selection / early stop)')
    parser.add_argument('--final-eval-episodes', type=int, default=300,
                        help='Episodes for the reported final eval of the BEST checkpoint')
    parser.add_argument('--min-rounds', type=int, default=8,
                        help='Do not early-stop before this many rounds')
    parser.add_argument('--patience', type=int, default=6,
                        help='Early-stop after this many rounds without improvement')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--out', default=None,
                        help='Explicit output JSON path (overrides the default '
                             'grid_asymmetric_<task>_... name). Used by the '
                             'comm-cost diagonal sweep so single-cell runs do '
                             'not collide.')
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = out_dir / (
            f"grid_asymmetric_{args.task}_{args.decay_regime}_{args.obs_regime}"
            f"_cf{int(args.c_fail_scale)}_seed{args.seed}.json"
        )

    results = run_grid_asymmetric(
        task_name=args.task,
        n_c_nar=args.n_c_nar,
        n_c_remind=args.n_c_remind,
        c_nar_min=args.c_nar_min,
        c_nar_max=args.c_nar_max,
        c_remind_min=args.c_remind_min,
        c_remind_max=args.c_remind_max,
        c_fail_scale=args.c_fail_scale,
        c_off_timing=args.c_off_timing,
        decay_regime=args.decay_regime,
        obs_regime=args.obs_regime,
        n_rounds=args.rounds,
        steps_per_round=args.steps,
        n_eval_episodes=args.final_eval_episodes,
        seed=args.seed,
        resume_path=args.resume,
        incremental_save_path=str(out_path),
        min_rounds=args.min_rounds,
        patience=args.patience,
        eval_episodes=args.eval_episodes,
    )
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
