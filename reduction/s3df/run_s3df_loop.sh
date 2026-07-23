#!/bin/bash
# Supervisor loop for hexabox reduction on S3DF.
# Submits 8h batch jobs repeatedly until all integrals are done.
# Mirrors training/nersc_perlmutter/train_loop.sh.
#
# Run from a screen session on sdfiana019 (NOT inside a job):
#   screen -S reduce-loop
#   bash reduction/s3df/run_s3df_loop.sh
#   Ctrl-A D to detach; `screen -r reduce-loop` to re-attach.
#
# Env switches:
#   MAX_SUBMISSIONS=N   safety cap (default 20)
#   SLEEP_BETWEEN=N     seconds between submissions (default 60)
#   INTEGRAL_LIST=path  (default reduction/integrals_to_reduce.txt)
#   OUTBASE=path        (default results/hexabox)

set -uo pipefail
cd "$(dirname "$0")/../.."

MAX_SUBMISSIONS=${MAX_SUBMISSIONS:-20}
SLEEP_BETWEEN=${SLEEP_BETWEEN:-60}
INTEGRAL_LIST=${INTEGRAL_LIST:-reduction/integrals_to_reduce.txt}
OUTBASE=${OUTBASE:-results/hexabox}

TOTAL=$(grep -c . "$INTEGRAL_LIST" || true)

mkdir -p logs
SUP_LOG=logs/reduce_supervisor_$(date +%Y%m%d_%H%M%S).log

log() { echo "[$(date -Iseconds)] $*" | tee -a "$SUP_LOG"; }

count_done() {
    find "$OUTBASE" -name "reduction.pkl" 2>/dev/null | wc -l | tr -d ' '
}

wait_for_job() {
    local jobid=$1
    log "Waiting for job $jobid to complete..."
    while squeue -j "$jobid" -h 2>/dev/null | grep -q "$jobid"; do
        sleep 30
    done
    log "Job $jobid finished."
}

log "Supervisor started. Total integrals: $TOTAL"
log "  MAX_SUBMISSIONS=$MAX_SUBMISSIONS"
log "  Supervisor log: $SUP_LOG"

for i in $(seq 1 $MAX_SUBMISSIONS); do
    done=$(count_done)
    log "--- submission $i/$MAX_SUBMISSIONS  done=$done/$TOTAL ---"

    if (( done >= TOTAL )); then
        log "All $TOTAL integrals complete. Stopping."
        exit 0
    fi

    log "Submitting sbatch job..."
    JOBID=$(sbatch --parsable reduction/s3df/allocation_s3df.sh)
    log "Submitted job $JOBID"

    wait_for_job "$JOBID"

    new_done=$(count_done)
    log "After job $JOBID: done=$new_done/$TOTAL (was $done, +$((new_done - done)))"

    if (( new_done == done )); then
        log "No progress — possible error. Check logs/reduce_${JOBID}.out"
        log "Stopping to avoid looping on a hard error."
        exit 1
    fi

    if (( i < MAX_SUBMISSIONS )); then
        log "Sleeping ${SLEEP_BETWEEN}s before next submission..."
        sleep "$SLEEP_BETWEEN"
    fi
done

log "Hit MAX_SUBMISSIONS. done=$(count_done)/$TOTAL."
exit 2
