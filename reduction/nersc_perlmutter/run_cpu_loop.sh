#!/bin/bash
# Supervisor loop for CPU reduction via the interactive queue.
# Mirrors training/nersc_perlmutter/train_loop.sh.
#
# Use this instead of sbatch when you want 4-hour interactive allocations
# with automatic restart (e.g. to monitor progress between chunks).
# For a single long run until maintenance, prefer:
#   sbatch reduction/nersc_perlmutter/allocation_cpu.sh
#
# Run from a tmux session on the LOGIN node (NOT inside an existing salloc):
#   tmux new -s reduce-loop
#   bash reduction/nersc_perlmutter/run_cpu_loop.sh
#   Ctrl-B D to detach; `tmux a -t reduce-loop` to re-attach.
#
# Env switches:
#   NODES=N              nodes per allocation (default 4)
#   WALLTIME=HH:MM:SS    per-allocation walltime (default 04:00:00)
#   MAX_ALLOCATIONS=N    safety cap (default 12 ≈ 48h total)
#   SLEEP_BETWEEN=N      seconds between allocations (default 30)
#   INTEGRAL_LIST=path   path to integral list (default reduction/integrals_to_reduce.txt)
#   OUTBASE=path         results root (default results/hexabox)
#   TOPOLOGY, MODEL      forwarded to allocation_cpu.sh

set -uo pipefail
cd "$(dirname "$0")/../.."

NODES=${NODES:-4}
WALLTIME=${WALLTIME:-04:00:00}
MAX_ALLOCATIONS=${MAX_ALLOCATIONS:-12}
SLEEP_BETWEEN=${SLEEP_BETWEEN:-30}
export INTEGRAL_LIST=${INTEGRAL_LIST:-reduction/integrals_to_reduce.txt}
export OUTBASE=${OUTBASE:-results/hexabox}
export TOPOLOGY=${TOPOLOGY:-topology_input/hexabox}
export MODEL=${MODEL:-checkpoints/hexabox_100k/best_model.pt}

TOTAL=$(grep -c . "$INTEGRAL_LIST" || true)

mkdir -p logs
SUP_LOG=logs/reduce_supervisor_$(date +%Y%m%d_%H%M%S).log

log() { echo "[$(date -Iseconds)] $*" | tee -a "$SUP_LOG"; }

count_done() {
    find "$OUTBASE" -name "reduction.pkl" 2>/dev/null | wc -l | tr -d ' '
}

log "Supervisor started. Total integrals: $TOTAL"
log "  NODES=$NODES  WALLTIME=$WALLTIME  MAX_ALLOCATIONS=$MAX_ALLOCATIONS"
log "  Supervisor log: $SUP_LOG"

for i in $(seq 1 $MAX_ALLOCATIONS); do
    done=$(count_done)
    log "--- iteration $i/$MAX_ALLOCATIONS  done=$done/$TOTAL ---"

    if (( done >= TOTAL )); then
        log "All $TOTAL integrals complete. Stopping."
        exit 0
    fi

    log "Requesting salloc -N $NODES -C cpu -q interactive -t $WALLTIME -A m4539 ..."
    log "(blocks until allocation is granted)"

    salloc \
        -N "$NODES" -C cpu -q interactive -t "$WALLTIME" -A m4539 \
        bash reduction/nersc_perlmutter/allocation_cpu.sh
    rc=$?
    log "salloc exited rc=$rc"

    new_done=$(count_done)
    log "After iteration: done=$new_done/$TOTAL (was $done, +$((new_done - done)))"

    if (( new_done == done )) && (( rc != 0 )); then
        log "No progress and non-zero exit — possible error. Stopping."
        log "Check $OUTBASE/logs/chunk_*.log for details."
        exit 1
    fi

    if (( i < MAX_ALLOCATIONS )); then
        log "Sleeping ${SLEEP_BETWEEN}s before next allocation..."
        sleep "$SLEEP_BETWEEN"
    fi
done

log "Hit MAX_ALLOCATIONS. done=$(count_done)/$TOTAL."
exit 2
