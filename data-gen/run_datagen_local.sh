#!/bin/bash
# ============================================================================
# SAILIR data-gen Stage 1 — LOCAL launcher (no Condor).
# Launches N workers as direct subprocesses, capped at MAX_CONCURRENT at once.
#
# Usage: bash data-gen/run_datagen_local.sh   (from SAILIR repo root)
#
# Edit the config block below, then run inside a screen session.
# ============================================================================
set -e

# ---- config: change these for your topology / dataset ----------------------
TOPOLOGY=topology_input/hexabox   # rel. to SAILIR_DIR; must have IBP, LI, masters
DATASET=hexabox_13M_biased         # output: data/${DATASET}_raw_jsonl/
N_WORKERS=1000                     # number of parallel workers (= number of shards)
N_SCRAMBLES=1000                   # scrambles per worker (1000x1000 = 1M scrambles ~ 13M samples)
MAX_CONCURRENT=120                 # max simultaneous workers (<=120 on tplx)
# ----------------------------------------------------------------------------

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAILIR_DIR="$BASE"
PYTHON="$BASE/venv/bin/python"
RAW_DIR="$BASE/data/${DATASET}_raw_jsonl"
LOGDIR="$BASE/data/${DATASET}_logs"

mkdir -p "$RAW_DIR" "$LOGDIR"

export SAILIR_DIR TOPOLOGY DATASET PYTHON

echo "============================================================"
echo "Stage 1: ${N_WORKERS} workers x ${N_SCRAMBLES} scrambles"
echo "  topology: ${TOPOLOGY}"
echo "  output:   ${RAW_DIR}"
echo "  logs:     ${LOGDIR}"
echo "  started:  $(date)"
echo "============================================================"

running=0
pids=()

for ((i=0; i<N_WORKERS; i++)); do
    # Throttle: wait for a slot if at MAX_CONCURRENT
    while (( running >= MAX_CONCURRENT )); do
        for idx in "${!pids[@]}"; do
            if ! kill -0 "${pids[$idx]}" 2>/dev/null; then
                wait "${pids[$idx]}" || true
                unset "pids[$idx]"
                running=$(( running - 1 ))
            fi
        done
        pids=("${pids[@]}")  # re-index to remove gaps
        sleep 1
    done

    bash "$BASE/data-gen/datagen_worker.sh" "$i" "$N_SCRAMBLES" "$RAW_DIR" \
        > "$LOGDIR/worker_${i}.out" 2>&1 &
    pids+=($!)
    running=$(( running + 1 ))
    echo "  launched worker $i (pid=${pids[-1]}, running=$running)"
done

echo "All ${N_WORKERS} workers launched. Waiting for completion..."
wait
echo ""
echo "============================================================"
echo "All workers done: $(date)"
echo "Shards written: $(ls "$RAW_DIR"/*.jsonl 2>/dev/null | wc -l) / ${N_WORKERS}"
echo "Total lines:    $(cat "$RAW_DIR"/*.jsonl 2>/dev/null | wc -l)"
echo "============================================================"
echo ""
echo "Next: run Stage 2 (preprocess to tensors):"
echo "  bash data-gen/run_preprocess_local.sh"
