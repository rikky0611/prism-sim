#!/usr/bin/env bash
# v5 first-iteration characterization: 7 tasks x 3 diagonal comm-cost cells.
# cn=cr in {0.5, 1.58, 5.0} (cheap / mid / expensive communication).
# Single seed; convergence is read from per-cell training_log learning curves.
# Sequential (avoid OOM). ~17h wall clock.
#
# Output: data/results/commsweep_<task>_c<cell>_seed0.json (one per task x cell)
#
# Usage:
#   nohup bash scripts/run_v5_commsweep.sh > data/logs/v5_commsweep_master.log 2>&1 &

set -uo pipefail
cd /home/ec2-user/prism-sim
mkdir -p data/logs data/results

ts() { date +'%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

# tasks ordered short -> long (make_cereal first = known-good sanity check)
TASKS=(make_cereal make_coffee make_sandwich make_tea cooking latte_making make_stencil)
CELLS=(0.5 1.58 5.0)

COMMON="--c-fail-scale 15 --obs-regime durable --rounds 10 --steps 50000 --eval-episodes 50 --seed 0"

log "=== v5 comm-sweep START: ${#TASKS[@]} tasks x ${#CELLS[@]} cells ==="
for task in "${TASKS[@]}"; do
    for c in "${CELLS[@]}"; do
        out="data/results/commsweep_${task}_c${c}_seed0.json"
        logf="data/logs/commsweep_${task}_c${c}_seed0.log"
        log "START task=$task cn=cr=$c"
        python3 -u src/experiments/run_grid_asymmetric.py \
            --task "$task" \
            --n-c-nar 1 --n-c-remind 1 \
            --c-nar-min "$c" --c-nar-max "$c" \
            --c-remind-min "$c" --c-remind-max "$c" \
            $COMMON --out "$out" > "$logf" 2>&1
        rc=$?
        if [ $rc -ne 0 ]; then
            log "WARN task=$task cn=cr=$c exited rc=$rc (continuing)"
        else
            log "DONE task=$task cn=cr=$c -> $out"
        fi
    done
done
log "=== v5 comm-sweep DONE ==="
