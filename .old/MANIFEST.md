# Archived Files - February 17, 2026

This directory contains files archived during project cleanup.

## Reason for Archival
These files were moved to reduce project clutter and avoid confusion when working on the extended training phase (50k → 200k timesteps).

## Contents
- **docs/** - Documentation (English, Japanese, papers)
- **data/results/*.json** - Old experiment results from 50k training
- **src/experiments/** - Old demo, test, and analysis scripts
  - analyze_episodes.py
  - demo_new_effectiveness.py
  - demo_recency_boost.py
  - test_recency_model.py
  - evaluate_cross_task_all_regimes.py
- **src/training/** - Old training script variants
  - train_cross_task_all_regimes.py
- **src/visualization/** - Old visualization scripts
  - visualize_kitchen.py
  - visualize_cross_task_multi_regime.py
  - generate_cross_task_presentation.py
  - generate_cross_task_report.py
- **cross_task_experiment_report.md** - Old experiment report

## Restoration
To restore any file:
```bash
cp -r .old/path/to/file original/location/
```

## Active Files (Kept in Project)

### Core Implementation
- `src/experiments/procedure_assistant_sim.py` - POMDP environment
- `src/experiments/task_definitions.py` - Task definitions
- `src/training/train_rl_policy.py` - PPO training core
- `src/training/train_cross_task_multi_regime.py` - Multi-regime training script
- `src/experiments/evaluate_cross_task_multi_regime.py` - Current evaluation script

### Current Visualization
- `src/visualization/visualize_episode_trajectory.py` - Trajectory visualization
- `src/visualization/visualize_training_curves.py` - Training curves
- `src/visualization/generate_results_presentation.py` - Current presentation generator
- `src/visualization/generate_all_trajectories.sh` - Batch trajectory generation
- `src/visualization/generate_example_animations.sh` - Animation generation

### Data & Models
- `models/` - All trained models (50k timesteps)
- `results/figures/` - Recent visualizations
- `results/presentations/` - Recent presentations
- `results/videos/` - Animation GIFs
