# Framework Validation — Figure Manifest

| LaTeX `\ref`           | File (in `results/figures/`)                      | Source experiment | Source data |
|------------------------|---------------------------------------------------|-------------------|-------------|
| `fig:phase-asymmetric` | `phase_asymmetric_make_cereal_seed0.png`          | E1: cost-asymmetry grid (6×6, c_nar × c_remind, c_fail=15) | `grid_asymmetric_make_cereal_step_transition_durable_cf15_seed0.json` |
| `fig:phase-asymmetric-baselines` | `phase_asymmetric_baselines_make_cereal_seed0.png` | E1 baseline overlay: Passive/Heuristic/MA-IPPO across same grid | (same JSON, after `backfill_grid_asymmetric_baselines.py`) |
| `fig:phase-cfail`      | `phase_diagram_make_cereal_e2.png`                | E2: existing 15×15 c_comm × c_fail grid | `grid_search_make_cereal_step_transition_durable.json` |
| `fig:frontier-slices`  | `frontier_slices_make_cereal.png`                 | E2 supplement: frontier slices at fixed c_fail | (same) |
| `fig:sensing-sweep`    | `sensing_sweep_make_cereal_seed0.png`             | E3: 4×4 (n_base × λ_n) grid | `sensing_grid_make_cereal_cf15_cc0.50_seed0.json` |
| `fig:sensing-remind`   | `sensing_remind_sweep_make_cereal_seed0.png`      | E3-v2: 4×4 (λ_n × c_remind) grid, n_base=0.4 fixed; tests reminder-activation hypothesis | `sensing_remind_grid_make_cereal_cf15_n0.40_seed0.json` |
| `fig:phase-asymmetric-seeds` | `phase_asymmetric_make_cereal_seeds_aggregated.png` | E1 multi-seed aggregate (median + IQR), 3 seeds | `grid_asymmetric_make_cereal_step_transition_durable_cf15_seed{0,1,2}.json` |
| `fig:sensing-remind-seeds` | `sensing_remind_sweep_make_cereal_seeds_aggregated.png` | E3-v2 multi-seed aggregate, 3 seeds | `sensing_remind_grid_make_cereal_cf15_n0.40_seed{0,1,2}.json` |
| `fig:multitask-phase`  | `phase_diagram_multitask.png`                     | E2 cross-task: side-by-side categorical phase diagrams (make_cereal/cooking/latte_making/make_stencil) | `grid_search_<task>_step_transition_durable.json` |
| `fig:pareto`           | `pareto_per_task.png` + `pareto_improvement_bars.png` | E4: cross-task baseline comparison (7 tasks × 3 obs × 5 fail × 3 policies × 500 episodes) | `comparison_3policy*.json` |
| `fig:trajectories`     | `trajectory_anatomy_make_cereal_seed0.png`        | E5: deterministic single-episode rollouts from 4 phase-corner cells of the E1 grid | trained models in `models/ma_ippo/make_cereal/asym_cn*_cr*_seed0/` |

## Reproduction commands

```
# E1 (asymmetric grid) — ~2 hours on 2 CPUs
python3 src/experiments/run_grid_asymmetric.py --task make_cereal \
    --n-c-nar 6 --n-c-remind 6 --c-fail-scale 15 \
    --rounds 4 --steps 10000 --eval-episodes 30 --seed 0

# E1 baselines (overlay) — ~30 seconds, no training
python3 src/experiments/backfill_grid_asymmetric_baselines.py \
    --results data/results/grid_asymmetric_make_cereal_step_transition_durable_cf15_seed0.json \
    --eval-episodes 30
python3 src/visualization/plot_phase_asymmetric_baselines.py \
    --results data/results/grid_asymmetric_make_cereal_step_transition_durable_cf15_seed0.json

# E3 (sensing grid) — ~1 hour on 2 CPUs
python3 src/experiments/run_sensing_grid.py --task make_cereal \
    --n-noise 4 --n-lambda 4 --rounds 4 --steps 10000 --eval-episodes 30 --seed 0

# E3-v2 (brittleness × reminder-cost) — ~1 hour on 2 CPUs
python3 src/experiments/run_sensing_remind_grid.py --task make_cereal \
    --n-lambda 4 --n-c-remind 4 --rounds 4 --steps 10000 --eval-episodes 30 --seed 0

# E2 cross-task grids (cooking, latte_making, make_stencil) — ~3.5 h each
for task in cooking latte_making make_stencil; do
    python3 src/experiments/run_grid_search.py --task $task \
        --n-comm 8 --n-fail 8 --rounds 4 --steps 10000 --eval-episodes 30
done

# Plotting (no compute)
#
# Every plot script below accepts a `--paper` flag that ALSO copies the
# rendered PNG to `results/figures/paper/`, the curated set referenced
# from the LaTeX paper. Drop `--paper` for exploratory runs.
python3 src/visualization/plot_phase_asymmetric.py            --results data/results/grid_asymmetric_make_cereal_step_transition_durable_cf15_seed0.json --paper
python3 src/visualization/plot_phase_asymmetric_baselines.py  --results data/results/grid_asymmetric_make_cereal_step_transition_durable_cf15_seed0.json --paper
python3 src/visualization/plot_phase_diagram.py               --results data/results/grid_search_make_cereal_step_transition_durable.json --output results/figures/phase_diagram_make_cereal_e2.png --paper
python3 src/visualization/plot_phase_multitask.py             --tasks make_cereal cooking latte_making make_stencil --paper
python3 src/visualization/plot_frontier_slices.py             --results data/results/grid_search_make_cereal_step_transition_durable.json --paper
python3 src/visualization/plot_sensing_sweep.py               --results data/results/sensing_grid_make_cereal_cf15_cc0.50_seed0.json --noise-slice 0.4 --paper
python3 src/visualization/plot_sensing_remind_sweep.py        --results data/results/sensing_remind_grid_make_cereal_cf15_n0.40_seed0.json --paper
python3 src/visualization/plot_phase_asymmetric_seeds.py      --results 'data/results/grid_asymmetric_make_cereal_step_transition_durable_cf15_seed*.json' --paper
python3 src/visualization/plot_sensing_remind_seeds.py        --results 'data/results/sensing_remind_grid_make_cereal_cf15_n0.40_seed*.json' --paper
python3 src/visualization/plot_pareto_per_task.py --paper
python3 src/visualization/plot_trajectory_anatomy.py --paper
```

## Headline numerical results

- E1: division-of-labor index ranges from 1.00 (cheap human, expensive assistant) to 0.00 (expensive human, cheap assistant); 3-seed median phase distribution Silent 19% / Human-led 36% / Assistant-led 25% / Mixed 19%; mean inter-seed DoL IQR = 0.058 on [0,1].
- E2 cross-task (8×8 grids): topology preserved on cooking / latte_making / make_stencil; Silent shrinks (37%→5%/6%/3%) and Mixed grows (5%→50%/36%/61%) as task length and criticality density increase; Assistant-led tracks the number of critical steps.
- E2: from existing 15×15 grid, behavior partitions into Silent 37.8% / Question 32.0% / Remind 22.7% / mixed 7.5% with sharp boundaries.
- E3: tracking accuracy 0.83 → 0.49 across n_base ∈ {0.2, …, 0.8}; confirm actions activate at the hardest cell (n_base=0.8, λ_n=0.20) ≈10/episode; reminders absent throughout this regime.
- E4: MA-IPPO mean reward ≥ best baseline in 79% of 105 (task × obs × fail) cells across 7 tasks; strict Pareto-dominance on (interactions, failures) is rare (0% vs Passive, 1% vs Heuristic) because the two baselines occupy opposite extremes — the framework's contribution is intermediate operating-point selection, not uniform axis improvement.
