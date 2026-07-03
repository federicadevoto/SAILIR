#!/bin/bash
# ============================================================================
# SAILIR data-gen Stage 2 — LOCAL launcher (no Condor).
# Converts raw JSONL shards into packed tensor shards (.pt), one process per
# shard, capped at MAX_CONCURRENT at once.
#
# Usage: bash data-gen/run_preprocess_local.sh   (from SAILIR repo root)
#
# Edit the config block below (must match the Stage-1 run), then run inside
# a screen session.
# ============================================================================
set -e

# ---- config: must match the Stage-1 run ------------------------------------
TOPOLOGY=topology_input/hexabox   # rel. to SAILIR_DIR
DATASET=hexabox_test               # input: data/${DATASET}_raw_jsonl/
N_SHARDS=50                        # = N_WORKERS from Stage 1
MAX_CONCURRENT=50                  # max simultaneous preprocess jobs
# ----------------------------------------------------------------------------

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$BASE/venv/bin/python"
RAW_DIR="$BASE/data/${DATASET}_raw_jsonl"
PACKED_DIR="$BASE/data/${DATASET}_packed"
LOGDIR="$BASE/data/${DATASET}_preprocess_logs"

mkdir -p "$PACKED_DIR" "$LOGDIR"

echo "============================================================"
echo "Stage 2: ${N_SHARDS} shards -> packed tensors"
echo "  topology: ${TOPOLOGY}"
echo "  input:    ${RAW_DIR}"
echo "  output:   ${PACKED_DIR}"
echo "  logs:     ${LOGDIR}"
echo "  started:  $(date)"
echo "============================================================"

running=0
pids=()

for ((i=0; i<N_SHARDS; i++)); do
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

    INPUT="$RAW_DIR/multisector_data_worker${i}.jsonl"
    OUT_DIR="$PACKED_DIR/shard_${i}"
    mkdir -p "$OUT_DIR"

    PYTHONUNBUFFERED=1 "$PYTHON" -u "$BASE/data-gen/preprocess_to_tensors.py" \
        --topology   "$BASE/$TOPOLOGY" \
        --input      "$INPUT" \
        --output_dir "$OUT_DIR" \
        --val_split 0.1 --test_split 0.1 --seed $(( 42 + i )) \
        > "$LOGDIR/shard_${i}.out" 2>&1 &
    pids+=($!)
    running=$(( running + 1 ))
    echo "  launched shard $i (pid=${pids[-1]}, running=$running)"
done

echo "All ${N_SHARDS} shards launched. Waiting for completion..."
wait
echo ""
echo "============================================================"
echo "All shards done: $(date)"
echo "Packed shards: $(ls -d "$PACKED_DIR"/shard_* 2>/dev/null | wc -l) / ${N_SHARDS}"
echo "============================================================"
echo ""
echo "Next: training"
echo "  see training/README.md"
