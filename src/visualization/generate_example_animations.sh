#!/bin/bash
# Generate example animated GIF visualizations for 5 representative episodes

echo "Generating example animation GIFs..."
echo ""

# 1. Best performer: make_sandwich / moderate_low (95.4% improvement)
echo "[1/5] Best performer: make_sandwich / moderate_low"
python3 visualize_episode_trajectory.py --task make_sandwich --regime moderate_low --animate 2>&1 | grep -E "(Complete|saved to)"

# 2. Strategic interventions: make_coffee / balanced (1.17 interventions)
echo ""
echo "[2/5] Strategic interventions: make_coffee / balanced"
python3 visualize_episode_trajectory.py --task make_coffee --regime balanced --animate 2>&1 | grep -E "(Complete|saved to)"

# 3. Pure silence: make_cereal / very_high_stakes (0 interventions)
echo ""
echo "[3/5] Pure silence: make_cereal / very_high_stakes"
python3 visualize_episode_trajectory.py --task make_cereal --regime very_high_stakes --animate 2>&1 | grep -E "(Complete|saved to)"

# 4. Anomaly: make_stencil / very_high_stakes (67 interventions!)
echo ""
echo "[4/5] Anomaly (over-intervention): make_stencil / very_high_stakes"
python3 visualize_episode_trajectory.py --task make_stencil --regime very_high_stakes --animate 2>&1 | grep -E "(Complete|saved to)"

# 5. Complex task: latte_making / balanced (20 steps, 0 interventions)
echo ""
echo "[5/5] Complex task: latte_making / balanced"
python3 visualize_episode_trajectory.py --task latte_making --regime balanced --animate 2>&1 | grep -E "(Complete|saved to)"

echo ""
echo "✓ All example animations generated!"
echo ""
echo "Output location: ../../results/videos/"
ls -lh ../../results/videos/trajectory_*.gif 2>/dev/null | wc -l
echo "GIF files created"
