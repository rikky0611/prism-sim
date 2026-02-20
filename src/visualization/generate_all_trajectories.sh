#!/bin/bash
# Generate trajectory visualizations for all 21 models

tasks=(make_cereal make_coffee make_tea make_sandwich make_stencil cooking latte_making)
regimes=(very_high_stakes balanced moderate_low)

echo "Generating trajectories for all 21 models..."
count=0
total=21

for task in "${tasks[@]}"; do
  for regime in "${regimes[@]}"; do
    count=$((count + 1))
    echo "[$count/$total] Generating $task / $regime"
    python3 visualize_episode_trajectory.py --task "$task" --regime "$regime" 2>&1 | grep -E "(Complete|Interventions|Failures|saved to)"
  done
done

echo ""
echo "✓ All trajectories generated!"
ls -lh ../../results/figures/trajectory_*.png | wc -l
