#!/usr/bin/env bash
# Overnight validation strengthening: Phase B (seeds) + Phase D (multi-task).
# Launches sequentially; safe to start after Phase C (E3-v2 seed 0) completes.
#
# Total expected wall-clock on 2 CPU / 3 GB RAM: ~14-16 hours.
#
# Usage:
#   nohup bash scripts/run_validation_overnight.sh > data/logs/overnight.log 2>&1 &

set -euo pipefail
cd /home/ec2-user/prism-sim
mkdir -p data/logs

ts() { date +'%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a data/logs/overnight.log; }

log "=== overnight start ==="

# ---------------------------------------------------------------------------
# Phase B-1: E1 seed 1 (~2.5h)
# ---------------------------------------------------------------------------
log "Phase B-1: E1 seed 1 (asymmetric grid 6x6)"
python3 -u src/experiments/run_grid_asymmetric.py --task make_cereal \
    --n-c-nar 6 --n-c-remind 6 --c-fail-scale 15 \
    --rounds 4 --steps 10000 --eval-episodes 30 --seed 1 \
    > data/logs/e1_seed1.log 2>&1
log "Phase B-1 done"

# ---------------------------------------------------------------------------
# Phase B-2: E1 seed 2 (~2.5h)
# ---------------------------------------------------------------------------
log "Phase B-2: E1 seed 2"
python3 -u src/experiments/run_grid_asymmetric.py --task make_cereal \
    --n-c-nar 6 --n-c-remind 6 --c-fail-scale 15 \
    --rounds 4 --steps 10000 --eval-episodes 30 --seed 2 \
    > data/logs/e1_seed2.log 2>&1
log "Phase B-2 done"

# ---------------------------------------------------------------------------
# Phase B-3: E3-v2 seed 1 (~1h)
# ---------------------------------------------------------------------------
log "Phase B-3: E3-v2 seed 1 (sensing-remind grid 4x4)"
python3 -u src/experiments/run_sensing_remind_grid.py --task make_cereal \
    --n-lambda 4 --n-c-remind 4 \
    --rounds 4 --steps 10000 --eval-episodes 30 --seed 1 \
    > data/logs/e3v2_seed1.log 2>&1
log "Phase B-3 done"

# ---------------------------------------------------------------------------
# Phase B-4: E3-v2 seed 2 (~1h)
# ---------------------------------------------------------------------------
log "Phase B-4: E3-v2 seed 2"
python3 -u src/experiments/run_sensing_remind_grid.py --task make_cereal \
    --n-lambda 4 --n-c-remind 4 \
    --rounds 4 --steps 10000 --eval-episodes 30 --seed 2 \
    > data/logs/e3v2_seed2.log 2>&1
log "Phase B-4 done"

# ---------------------------------------------------------------------------
# Phase D-1: cooking c_comm × c_fail grid 8x8 (~3.5h)
# ---------------------------------------------------------------------------
log "Phase D-1: grid_search cooking 8x8"
python3 -u src/experiments/run_grid_search.py --task cooking \
    --n-comm 8 --n-fail 8 --rounds 4 --steps 10000 --eval-episodes 30 \
    > data/logs/d_cooking.log 2>&1
log "Phase D-1 done"

# ---------------------------------------------------------------------------
# Phase D-2: latte_making 8x8 (~3.5h)
# ---------------------------------------------------------------------------
log "Phase D-2: grid_search latte_making 8x8"
python3 -u src/experiments/run_grid_search.py --task latte_making \
    --n-comm 8 --n-fail 8 --rounds 4 --steps 10000 --eval-episodes 30 \
    > data/logs/d_latte.log 2>&1
log "Phase D-2 done"

# ---------------------------------------------------------------------------
# Phase D-3: make_stencil 8x8 (~3.5h)
# ---------------------------------------------------------------------------
log "Phase D-3: grid_search make_stencil 8x8"
python3 -u src/experiments/run_grid_search.py --task make_stencil \
    --n-comm 8 --n-fail 8 --rounds 4 --steps 10000 --eval-episodes 30 \
    > data/logs/d_stencil.log 2>&1
log "Phase D-3 done"

log "=== overnight finished ==="
