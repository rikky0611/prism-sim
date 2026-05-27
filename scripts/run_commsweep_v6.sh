#!/usr/bin/env bash
# v6 comm-cost sweep: 7 tasks x 3 communication-cost cells, NEW protocol
#   - v5 simulator (identity-based, critical=-50 terminal, path-aware off-timing)
#   - best-checkpoint reporting + early stopping (max 20 rounds, patience 6, min 8)
#   - per-round eval 100, final eval 300 (low variance)
# Sequential (parallel OOM-killed previously). Incremental save per cell.
#
# Usage:
#   nohup bash scripts/run_commsweep_v6.sh > data/logs/commsweep_v6_master.log 2>&1 &

set -uo pipefail
cd /home/ec2-user/prism-sim
mkdir -p data/logs data/results

ts() { date +'%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

TASKS=(make_coffee make_cereal make_sandwich make_tea cooking latte_making make_stencil)
CVALS=(0.5 1.58 5.0)

COMMON="--n-c-nar 1 --n-c-remind 1 --obs-regime durable \
  --rounds 20 --min-rounds 8 --patience 6 \
  --eval-episodes 100 --final-eval-episodes 300 --steps 50000 --seed 0"

log "=== commsweep_v6 START (7 tasks x 3 cells, best-proto) ==="
n=0; total=$(( ${#TASKS[@]} * ${#CVALS[@]} ))
for task in "${TASKS[@]}"; do
  for c in "${CVALS[@]}"; do
    n=$((n+1))
    out="data/results/commsweep_v6_${task}_c${c}_seed0.json"
    lg="data/logs/commsweep_v6_${task}_c${c}_seed0.log"
    if [ -f "$out" ]; then
      log "[$n/$total] SKIP (exists) task=$task c=$c"
      continue
    fi
    log "[$n/$total] START task=$task c=$c"
    python3 -u src/experiments/run_grid_asymmetric.py --task "$task" \
      --c-nar-min "$c" --c-nar-max "$c" --c-remind-min "$c" --c-remind-max "$c" \
      $COMMON --out "$out" > "$lg" 2>&1
    rc=$?
    log "[$n/$total] DONE task=$task c=$c rc=$rc"
  done
done
log "=== commsweep_v6 DONE ==="
