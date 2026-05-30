#!/usr/bin/env bash
# 2D sweep validation for ML-1 (memory_estimate ground-truth leak) and T4
# (same-tick reminder failure-prob exploit) fixes. Re-runs the same 9-cell
# (acc x comm) grid on make_cereal and latte_making, output paths identical
# to scripts/run_sweep2d.sh so we can compare against the backed-up
# preML1T4 results in data/results/_backup_preML1T4/.
#
# Same v6 protocol as run_sweep2d.sh / run_sweep2d_remaining.sh:
#   --rounds 20 --min-rounds 8 --patience 6 --eval-episodes 100
#   --final-eval-episodes 300 --steps 50000 --seed 0
#
# Usage:
#   nohup bash scripts/run_sweep2d_ml1t4_validation.sh > data/logs/sweep2d_ml1t4_master.log 2>&1 &

set -uo pipefail
cd /home/ec2-user/prism-sim
mkdir -p data/logs data/results

ts() { date +'%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

TASKS=(make_cereal latte_making)
ACCS=(acc50 acc75 acc100)
CVALS=(0.5 1.58 5.0)

COMMON="--n-c-nar 1 --n-c-remind 1 --rounds 20 --min-rounds 8 --patience 6 \
  --eval-episodes 100 --final-eval-episodes 300 --steps 50000 --seed 0"

n=0; total=$(( ${#TASKS[@]} * ${#ACCS[@]} * ${#CVALS[@]} ))
log "=== sweep2d ML1+T4 VALIDATION START (${#TASKS[@]} tasks x 3 acc x 3 comm = $total cells) ==="
for task in "${TASKS[@]}"; do
  for acc in "${ACCS[@]}"; do
    for c in "${CVALS[@]}"; do
      n=$((n+1))
      out="data/results/sweep2d_${task}_${acc}_c${c}_seed0.json"
      lg="data/logs/sweep2d_ml1t4_${task}_${acc}_c${c}_seed0.log"
      if [ -f "$out" ]; then
        log "[$n/$total] SKIP (exists) task=$task acc=$acc c=$c"; continue
      fi
      log "[$n/$total] START task=$task acc=$acc c=$c"
      python3 -u src/experiments/run_grid_asymmetric.py --task "$task" \
        --c-nar-min "$c" --c-nar-max "$c" --c-remind-min "$c" --c-remind-max "$c" \
        --obs-regime "$acc" $COMMON --out "$out" > "$lg" 2>&1
      log "[$n/$total] DONE task=$task acc=$acc c=$c rc=$?"
    done
  done
done
log "=== sweep2d ML1+T4 VALIDATION DONE ==="
