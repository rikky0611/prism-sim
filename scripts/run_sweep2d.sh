#!/usr/bin/env bash
# 2D sweep: frame accuracy x communication cost, for 2 representative tasks.
#   accuracy axis: acc50 / acc75 / acc100  (frame recognizer accuracy 0.5/0.75/1.0)
#   comm-cost axis: c_nar=c_remind in {0.5, 1.58, 5.0}
#   tasks: make_cereal (8 steps, 2 critical), cooking (14 steps, 4 critical)
# v6 protocol: best-checkpoint reporting + early stopping (max 20, min 8,
#   patience 6), per-round eval 100, final eval 300, obs_noise_model='uniform'.
# Sequential; per-cell incremental save; SKIP-if-exists (resumable).
#
# Usage:
#   nohup bash scripts/run_sweep2d.sh > data/logs/sweep2d_master.log 2>&1 &

set -uo pipefail
cd /home/ec2-user/prism-sim
mkdir -p data/logs data/results

ts() { date +'%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

TASKS=(make_cereal cooking)
ACCS=(acc50 acc75 acc100)
CVALS=(0.5 1.58 5.0)

COMMON="--n-c-nar 1 --n-c-remind 1 --rounds 20 --min-rounds 8 --patience 6 \
  --eval-episodes 100 --final-eval-episodes 300 --steps 50000 --seed 0"

n=0; total=$(( ${#TASKS[@]} * ${#ACCS[@]} * ${#CVALS[@]} ))
log "=== sweep2d START (${#TASKS[@]} tasks x 3 acc x 3 comm = $total cells) ==="
for task in "${TASKS[@]}"; do
  for acc in "${ACCS[@]}"; do
    for c in "${CVALS[@]}"; do
      n=$((n+1))
      out="data/results/sweep2d_${task}_${acc}_c${c}_seed0.json"
      lg="data/logs/sweep2d_${task}_${acc}_c${c}_seed0.log"
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
log "=== sweep2d DONE ==="
