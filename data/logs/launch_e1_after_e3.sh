#!/bin/bash
# Wait for E3 to finish, then launch E1 (cost-asymmetry grid).
# Polls every 30 seconds.
E3_PID=9228
LOG_E1=/home/ec2-user/prism-sim/data/logs/e1_grid_asymmetric.log

# Wait for E3 to finish
while kill -0 "$E3_PID" 2>/dev/null; do
    sleep 30
done

cd /home/ec2-user/prism-sim/src/experiments
nohup python3 -u run_grid_asymmetric.py \
    --task make_cereal --n-c-nar 6 --n-c-remind 6 \
    --c-fail-scale 15 --rounds 4 --steps 10000 --eval-episodes 30 --seed 0 \
    > "$LOG_E1" 2>&1
