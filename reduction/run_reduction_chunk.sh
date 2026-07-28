#!/bin/bash
# Processes a subset of integrals: line n (0-indexed) goes to chunk (n % N_CHUNKS).
#
# Required env vars (set by nersc_perlmutter/srun_chunk.sh or a test wrapper):
#   CHUNK_ID      0-based index of this chunk
#   N_CHUNKS      total number of chunks
#   MAX_CPUS      CPU budget for hierarchical_reduction.py
#
# Optional overrides (all default to paths relative to repo root):
#   INTEGRAL_LIST, MODEL, TOPOLOGY, OUTBASE, PYTHON
#   INTEGRAL_TIMEOUT   wall-clock seconds per integral (default 10800 = 3h)
#                      on timeout, writes reduction.timeout and skips to next

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INTEGRAL_LIST=${INTEGRAL_LIST:-$BASE/reduction/integrals_to_reduce.txt}
MODEL=${MODEL:-$BASE/checkpoints/hexabox_100k/best_model.pt}
TOPOLOGY=${TOPOLOGY:-$BASE/topology_input/hexabox}
OUTBASE=${OUTBASE:-$BASE/results/hexabox}
PYTHON=${PYTHON:-python}
MAX_CPUS=${MAX_CPUS:-8}
CHUNK_ID=${CHUNK_ID:-0}
N_CHUNKS=${N_CHUNKS:-1}
INTEGRAL_TIMEOUT=${INTEGRAL_TIMEOUT:-10800}

# Extract this chunk's integrals: line index (0-based) % N_CHUNKS == CHUNK_ID
CHUNK_FILE=$(mktemp /tmp/sailir_chunk_${CHUNK_ID}_XXXXXX.txt)
trap "rm -f '$CHUNK_FILE'" EXIT

awk -v cid="$CHUNK_ID" -v nc="$N_CHUNKS" \
    'NF && !/^#/ { if (((NR-1) % nc) == cid) print }' \
    "$INTEGRAL_LIST" > "$CHUNK_FILE"

N_INTEGRALS=$(wc -l < "$CHUNK_FILE" | tr -d ' ')
echo "[chunk $CHUNK_ID/$N_CHUNKS] $N_INTEGRALS integrals  MAX_CPUS=$MAX_CPUS  pid=$$"
mkdir -p "$OUTBASE/logs"

n=0
while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue

    INTEGRAL_STR=$(echo "$line" | sed 's/TB\[//;s/\]//')
    LABEL=$(echo "$INTEGRAL_STR" | tr ',' '_')
    OUTDIR=$OUTBASE/$LABEL

    # skip if already done or previously timed out
    if [[ -f "$OUTDIR/reduction.pkl" || -f "$OUTDIR/reduction.timeout" ]]; then
        echo "[chunk $CHUNK_ID] skipping $((++n))/$N_INTEGRALS: TB[$INTEGRAL_STR] (already done/timed out)"
        continue
    fi

    mkdir -p "$OUTDIR/logs" "$OUTDIR/work"
    echo "[chunk $CHUNK_ID] integral $((++n))/$N_INTEGRALS: TB[$INTEGRAL_STR]"

    PYTHONUNBUFFERED=1 timeout "$INTEGRAL_TIMEOUT" \
        $PYTHON -u "$BASE/reduction/hierarchical_reduction.py" \
        --topology         "$TOPOLOGY" \
        --integral="$INTEGRAL_STR" \
        --output           "$OUTDIR/reduction.pkl" \
        --work-dir         "$OUTDIR/work" \
        --model-checkpoint "$MODEL" \
        --beam_width 40 --max_steps 1000000 --prime 1009 \
        --no-paper-masters-only \
        --use-v7-worker \
        --v7-cpus 1 \
        --backend local \
        --max-cpus        "$MAX_CPUS" \
        --straggler-timeout 1000000000 \
        --straggler2-timeout 1000000000 \
        --check-interval 5 \
        --max-concurrent 1000 \
        --resume \
        2>&1 | tee "$OUTDIR/logs/hierarchical.log"
    PIPE_RC=${PIPESTATUS[0]}

    if [[ $PIPE_RC -eq 124 ]]; then
        echo "[chunk $CHUNK_ID] TIMEOUT (${INTEGRAL_TIMEOUT}s) on integral $n: TB[$INTEGRAL_STR]"
        touch "$OUTDIR/reduction.timeout"
    elif [[ $PIPE_RC -ne 0 && ! -f "$OUTDIR/reduction.pkl" ]]; then
        echo "[chunk $CHUNK_ID] FAILED (rc=$PIPE_RC) integral $n: TB[$INTEGRAL_STR]"
    fi

    echo "[chunk $CHUNK_ID] done integral $n at $(date)"
done < "$CHUNK_FILE"

echo "[chunk $CHUNK_ID/$N_CHUNKS] all $N_INTEGRALS integrals done: $(date)"
