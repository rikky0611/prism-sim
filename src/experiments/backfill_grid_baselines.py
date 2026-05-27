"""
Backfill baseline metrics into an existing grid_search_*.json.

For each grid cell, reconstructs the MA environment at (c_fail_scale, c_comm)
using the top-level regime settings stored in the JSON, then rolls out the
"both-silent" baseline (PassiveHuman + SilentAssistant = no system, no proactive
human communication) and writes `baseline_mean_reward` / `baseline_mean_failures`
back into the cell.

No training happens here — this only evaluates fixed policies in the env, so
it's cheap enough to run over an existing 10x10 grid in minutes.

Usage:
    python backfill_grid_baselines.py \
        --results data/results/grid_search_make_cereal_step_transition_durable.json \
        [--eval-episodes 50] [--force]

Idempotent: cells that already have a non-null baseline_mean_reward are skipped
unless --force is passed.
"""

import sys
import json
import argparse
import shutil
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training"))

from task_definitions import load_task_definitions
from regime_definitions import build_params_grid
from ma_procedure_assistant_sim import (
    MAProcedureAssistantEnv,
    PassiveHumanPolicy,
    SilentAssistantPolicy,
)
from run_ma_experiments import evaluate_baseline_joint


def backfill(results_path: str, eval_episodes: int = None, force: bool = False,
             make_backup: bool = True) -> None:
    path = Path(results_path)
    with open(path) as f:
        d = json.load(f)

    task_name    = d['task']
    decay_regime = d.get('decay_regime', 'step_transition')
    obs_regime   = d.get('obs_regime', 'durable')
    c_remind     = d.get('c_remind', 1.0)
    c_confirm    = d.get('c_confirm', 1.0)
    c_off_timing = d.get('c_off_timing', 3.0)
    n_eval = eval_episodes if eval_episodes is not None else d.get('n_eval_episodes', 50)

    tasks = load_task_definitions()
    if task_name not in tasks:
        raise ValueError(f"Unknown task '{task_name}' (available: {list(tasks.keys())})")
    task_def = tasks[task_name]

    # Count work
    cells = [cell for row in d['grid'] for cell in row if cell is not None]
    total = len(cells)
    todo = [c for c in cells
            if force or c.get('baseline_mean_reward') is None]

    print(f"Backfilling baselines for: {path.name}")
    print(f"  task={task_name}  decay={decay_regime}  obs={obs_regime}")
    print(f"  c_remind={c_remind}  c_confirm={c_confirm}  c_off_timing={c_off_timing}")
    print(f"  n_eval_episodes={n_eval}")
    print(f"  cells: {len(todo)} to process / {total} total "
          f"(skipping {total - len(todo)} already done)")

    if not todo:
        print("Nothing to do.")
        return

    if make_backup:
        bak = path.with_suffix(path.suffix + '.bak')
        if not bak.exists():
            shutil.copy2(path, bak)
            print(f"  backup: {bak.name}")

    t0_total = time.time()
    for i, cell in enumerate(todo, 1):
        t0 = time.time()
        c_fail = cell['c_fail_scale']
        c_comm = cell['c_comm']

        params = build_params_grid(
            task_def, c_fail, c_comm,
            c_remind=c_remind, c_confirm=c_confirm,
            c_off_timing=c_off_timing,
            decay_regime=decay_regime, obs_regime=obs_regime,
        )
        ma_env = MAProcedureAssistantEnv(params, task_def)

        bm = evaluate_baseline_joint(
            PassiveHumanPolicy(), SilentAssistantPolicy(),
            ma_env, n_eval,
        )

        cell['baseline_mean_reward']   = bm['mean_reward']
        cell['baseline_mean_failures'] = bm['mean_failures']

        elapsed = time.time() - t0
        eta = (time.time() - t0_total) / i * (len(todo) - i)
        rl_r = cell.get('mean_reward', 0.0)
        print(f"[{i:3d}/{len(todo)}] cf={c_fail:6.2f} cc={c_comm:.4f} | "
              f"baseline r={bm['mean_reward']:+.2f} fail={bm['mean_failures']:.2f} | "
              f"Δ(RL-b)={rl_r - bm['mean_reward']:+.2f} | "
              f"{elapsed:.1f}s  ETA {eta/60:.1f}min")

    # Persist
    with open(path, 'w') as f:
        json.dump(d, f, indent=2)
    print(f"\nSaved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description='Backfill "both-silent" baseline metrics into grid_search_*.json'
    )
    parser.add_argument('--results', required=True,
                        help='Path to existing grid_search_*.json')
    parser.add_argument('--eval-episodes', type=int, default=None,
                        help='Episodes per baseline eval (default: reuse n_eval_episodes from JSON)')
    parser.add_argument('--force', action='store_true',
                        help='Re-evaluate cells that already have baseline_mean_reward')
    parser.add_argument('--no-backup', action='store_true',
                        help='Skip creating a .bak copy before overwriting')
    args = parser.parse_args()

    backfill(
        results_path=args.results,
        eval_episodes=args.eval_episodes,
        force=args.force,
        make_backup=not args.no_backup,
    )


if __name__ == '__main__':
    main()
