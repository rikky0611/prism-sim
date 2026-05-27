#!/usr/bin/env bash
# 4×4 H6 grid campaign across {make_cereal, latte_making} × {perfect, durable}.
# Sequential to avoid OOM. ~46h wall clock total.
#
# Order:
#   1. make_cereal perfect
#   2. (backup perfect models) make_cereal durable
#   3. latte_making perfect
#   4. (backup perfect models) latte_making durable
#
# Usage:
#   nohup bash scripts/run_h6_4x4_campaign.sh > data/logs/h6_4x4_campaign_master.log 2>&1 &

set -euo pipefail
cd /home/ec2-user/prism-sim
mkdir -p data/logs

ts() { date +'%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

GRID_ARGS="--n-c-nar 4 --n-c-remind 4 --c-fail-scale 15 --rounds 10 --steps 50000 --eval-episodes 50 --seed 0"

run() {
    local task="$1"
    local obs="$2"
    local logf="data/logs/grid_h6_4x4_${task}_${obs}_seed0.log"
    log "START  task=$task  obs=$obs  log=$logf"
    python3 -u src/experiments/run_grid_asymmetric.py \
        --task "$task" --obs-regime "$obs" $GRID_ARGS \
        > "$logf" 2>&1
    log "DONE   task=$task  obs=$obs"
}

backup_models() {
    local task="$1"
    local tag="$2"
    local src="models/ma_ippo/$task"
    local dst="models/ma_ippo/${task}.4x4_${tag}"
    log "BACKUP $src -> $dst"
    mkdir -p "$dst"
    # Move only the asym_*_seed0 directories produced by this run
    if compgen -G "$src/asym_*_seed0" > /dev/null; then
        mv "$src"/asym_*_seed0 "$dst"/ || true
    fi
}

log "=== H6 4x4 campaign START ==="

# 1. make_cereal perfect
run "make_cereal" "perfect"
backup_models "make_cereal" "perfect"

# 2. make_cereal durable
run "make_cereal" "durable"
backup_models "make_cereal" "durable"

# 3. latte_making perfect
run "latte_making" "perfect"
backup_models "latte_making" "perfect"

# 4. latte_making durable
run "latte_making" "durable"
backup_models "latte_making" "durable"

log "=== H6 4x4 campaign DONE ==="
