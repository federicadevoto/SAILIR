#!/bin/bash
# Supervisor loop: keeps requesting 4-hour interactive allocations and
# re-launching pentagonbox_10x training until TARGET_EPOCHS is reached.
#
# Run from a tmux session on the LOGIN node (NOT inside an existing salloc):
#
#   tmux new -s sailir-loop
#   bash training/nersc_perlmutter/train_loop.sh
#   # Ctrl-B D to detach
#
# Mechanics:
#   - Reads checkpoints/pentagonbox_10x/last.pt to find the last completed
#     epoch (key 'epoch').
#   - If current_epoch >= TARGET_EPOCHS, exits success.
#   - Otherwise requests a fresh `salloc -N 4 --gpus 16 -q interactive
#     -t 04:00:00` and runs allocation.sh inside it. The
#     launcher passes --auto_resume so training picks up at epoch+1.
#   - When the allocation ends (either training completed, or SLURM killed
#     it at 4h), the supervisor returns to the top of the loop.
#   - If training crashed before saving any new checkpoint AND exited
#     non-zero, the supervisor stops to avoid looping on a hard error.
#
# Env switches:
#   TARGET_EPOCHS=N    target total epochs (default 20)
#   MAX_ALLOCATIONS=N  safety cap on allocations (default 6 — covers ~24h
#                      wall clock if each runs the full 4 hours)
#   SLEEP_BETWEEN=N    seconds between allocations (default 30)
#   WALLTIME=HH:MM:SS  per-allocation walltime (default 04:00:00, the
#                      interactive-queue cap). Shorten for testing the
#                      supervisor itself (e.g. WALLTIME=00:05:00).
#   NODES=N            nodes per allocation (default 4)
#   GPUS=N             total GPUs per allocation (default 16, i.e. 4/node)
#   OUTPUT_DIR=path    checkpoint dir to watch (default checkpoints/pentagonbox_10x).
#                      Override when testing against a non-default run.

set -uo pipefail
cd "$(dirname "$0")/../.."

TARGET_EPOCHS=${TARGET_EPOCHS:-20}
MAX_ALLOCATIONS=${MAX_ALLOCATIONS:-6}
SLEEP_BETWEEN=${SLEEP_BETWEEN:-30}
WALLTIME=${WALLTIME:-04:00:00}
NODES=${NODES:-4}
GPUS=${GPUS:-16}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/pentagonbox_10x}
# Pass TOPOLOGY/SHARDS_DIR through to allocation.sh if set.
export TOPOLOGY SHARDS_DIR

mkdir -p logs
SUP_LOG=logs/supervisor_$(date +%Y%m%d_%H%M%S).log

log() {
    echo "[$(date -Iseconds)] $*" | tee -a "$SUP_LOG"
}

# pytorch module gives us a torch we can use for `torch.load` on CPU at the
# login node. Loading it once here so the helper below doesn't re-init it.
module load pytorch/2.11.0 2>/dev/null

get_current_epoch() {
    [[ -f "$OUTPUT_DIR/last.pt" ]] || { echo 0; return; }
    python - <<EOF 2>/dev/null
import torch
try:
    ck = torch.load("$OUTPUT_DIR/last.pt", map_location='cpu', weights_only=False)
    print(int(ck.get('epoch', 0)))
except Exception:
    print(0)
EOF
}

log "Supervisor started."
log "  TARGET_EPOCHS=$TARGET_EPOCHS  MAX_ALLOCATIONS=$MAX_ALLOCATIONS"
log "  Allocation:   -N $NODES --gpus $GPUS -t $WALLTIME"
log "  Output dir:   $OUTPUT_DIR"
log "  Supervisor log: $SUP_LOG (training logs are separate)"

for i in $(seq 1 $MAX_ALLOCATIONS); do
    current=$(get_current_epoch)
    log "--- iteration $i/$MAX_ALLOCATIONS  current_epoch=$current/$TARGET_EPOCHS ---"

    if (( current >= TARGET_EPOCHS )); then
        log "Target reached. Stopping."
        exit 0
    fi

    log "Requesting salloc -N $NODES --gpus $GPUS -q interactive -t $WALLTIME -A m4539_g ..."
    log "(this blocks until the allocation is granted; can take a few minutes)"

    # `salloc <cmd>` runs the command non-interactively inside the
    # allocation; salloc exits when the command exits (or SLURM kills it).
    # EPOCHS is propagated to the launcher, which passes it through as
    # --epochs $EPOCHS to train_classifier.py.
    EPOCHS=$TARGET_EPOCHS salloc \
        -N "$NODES" -C gpu --gpus "$GPUS" -q interactive -t "$WALLTIME" -A m4539_g \
        bash training/nersc_perlmutter/allocation.sh
    rc=$?
    log "salloc exited rc=$rc"

    new=$(get_current_epoch)
    log "After iteration: epoch=$new (was $current)"

    if (( new == current )) && (( rc != 0 )); then
        log "No progress and non-zero exit — likely a training error. Stopping."
        log "Investigate logs/pentagonbox_10x_full_*.log before relaunching."
        exit 1
    fi

    if (( i < MAX_ALLOCATIONS )); then
        log "Sleeping ${SLEEP_BETWEEN}s before next allocation..."
        sleep "$SLEEP_BETWEEN"
    fi
done

log "Hit MAX_ALLOCATIONS without reaching target. Last epoch=$(get_current_epoch)."
exit 2
