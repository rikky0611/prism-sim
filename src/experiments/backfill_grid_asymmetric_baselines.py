"""
Backfill Passive and Heuristic baselines into an existing
grid_asymmetric_*.json. Each cell is re-evaluated under the same
(c_nar, c_remind) costs as MA-IPPO, with two fixed-policy pairs:

  passive:   PassiveHuman + SilentAssistant
  heuristic: PassiveHuman + HeuristicReminderAssistant

Writes per-cell baseline metrics back into the JSON under keys
`passive_*` and `heuristic_*` (mean_reward, mean_failures, mean_reminds,
mean_narrations, mean_questions, mean_confirms, mean_tracking_map_acc).

No training. ~30 episodes/cell × 2 baselines × 36 cells ≈ a few minutes.

Usage:
    python backfill_grid_asymmetric_baselines.py \
        --results data/results/grid_asymmetric_make_cereal_step_transition_durable_cf15_seed0.json \
        [--eval-episodes 30] [--force]
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
from regime_definitions import build_params_asymmetric
from ma_procedure_assistant_sim import (
    MAProcedureAssistantEnv,
    PassiveHumanPolicy,
    SilentAssistantPolicy,
    HeuristicReminderAssistantPolicy,
)
from run_ma_experiments import evaluate_baseline_joint


def backfill(results_path: str, eval_episodes: int = 30, force: bool = False,
             make_backup: bool = True) -> None:
    path = Path(results_path)
    with open(path) as f:
        d = json.load(f)

    task_name    = d['task']
    decay_regime = d.get('decay_regime', 'step_transition')
    obs_regime   = d.get('obs_regime', 'durable')
    c_fail_scale = d.get('c_fail_scale', 15.0)
    c_off        = d.get('c_off_timing', 3.0)

    tasks = load_task_definitions()
    task_def = tasks[task_name]

    if make_backup:
        bak = path.with_suffix(path.suffix + '.bak')
        if not bak.exists():
            shutil.copy(path, bak)
            print(f"Backup: {bak}")

    n_total = sum(1 for row in d['grid'] for cell in row if cell is not None)
    n_done  = sum(1 for row in d['grid'] for cell in row
                  if cell is not None and cell.get('passive_mean_reward') is not None
                  and not force)
    print(f"Backfilling baselines into {n_total} cells "
          f"({n_done} already populated, will{'' if force else ' not'} re-eval)")
    t0 = time.time()
    n_updated = 0

    for row in d['grid']:
        for cell in row:
            if cell is None:
                continue
            if (not force) and cell.get('passive_mean_reward') is not None:
                continue

            params = build_params_asymmetric(
                task_def,
                c_fail_scale=c_fail_scale,
                c_nar=float(cell['c_nar']),
                c_remind=float(cell['c_remind']),
                c_off_timing=c_off,
                decay_regime=decay_regime,
                obs_regime=obs_regime,
            )
            ma_env = MAProcedureAssistantEnv(params, task_def)

            passive_h = PassiveHumanPolicy()
            silent_a  = SilentAssistantPolicy()
            heur_a    = HeuristicReminderAssistantPolicy(
                n_steps=ma_env.n_steps,
                critical_steps=ma_env.critical_steps,
            )

            mp = evaluate_baseline_joint(passive_h, silent_a, ma_env, n_episodes=eval_episodes)
            mh = evaluate_baseline_joint(passive_h, heur_a,   ma_env, n_episodes=eval_episodes)

            for k, v in mp.items():
                cell[f'passive_{k}'] = v
            for k, v in mh.items():
                cell[f'heuristic_{k}'] = v

            n_updated += 1
            if n_updated % 5 == 0 or n_updated == n_total:
                elapsed = time.time() - t0
                print(f"  cell {n_updated}/{n_total - n_done}  "
                      f"c_nar={cell['c_nar']:.3f} c_remind={cell['c_remind']:.3f}  "
                      f"passive_R={mp['mean_reward']:.1f} heur_R={mh['mean_reward']:.1f}  "
                      f"({elapsed:.1f}s)")

    with open(path, 'w') as f:
        json.dump(d, f, indent=2)
    print(f"Wrote {n_updated} updates to {path}  ({time.time()-t0:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--eval-episodes', type=int, default=30)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    backfill(args.results, args.eval_episodes, args.force)


if __name__ == '__main__':
    main()
