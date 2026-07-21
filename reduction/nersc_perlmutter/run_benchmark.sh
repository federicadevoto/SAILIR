#!/bin/bash
# Timing benchmark: reduces 5 integrals spanning r+s=6..13 sequentially.
# Meant to run inside a 30-min interactive CPU allocation so we can estimate
# throughput before committing to the full 700-integral production run.
#
# Usage (from repo root on Perlmutter):
#   salloc -N 1 -C cpu -q interactive -t 00:30:00 -A m4539
#   bash reduction/nersc_perlmutter/run_benchmark.sh
#
# Output: results/hexabox_bench/<label>/  and a timing summary at the end.

set -uo pipefail
cd "$(dirname "$0")/../.."

module load pytorch/2.11.0

TOPOLOGY=${TOPOLOGY:-topology_input/hexabox}
MODEL=${MODEL:-checkpoints/hexabox_100k/best_model.pt}
OUTBASE=${OUTBASE:-results/hexabox_bench}
INTEGRAL_LIST=${INTEGRAL_LIST:-reduction/benchmark_integrals.txt}
MAX_CPUS=${MAX_CPUS:-8}
PYTHON=${PYTHON:-python}

export PYTHONPATH="$(pwd)/sailir:$(pwd):${PYTHONPATH:-}"
mkdir -p "$OUTBASE/logs"

echo "================================================================"
echo "Hexabox reduction benchmark"
echo "  model:    $MODEL"
echo "  MAX_CPUS: $MAX_CPUS"
echo "  started:  $(date)"
echo "================================================================"
echo ""

declare -a RESULTS

n=0
while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue

    INTEGRAL_STR="$line"
    idx=(${line//,/ })
    r=0; s=0
    for x in "${idx[@]}"; do
        if (( x > 0 )); then (( r += x )); fi
        if (( x < 0 )); then (( s += -x )); fi
    done
    rs=$((r + s))

    LABEL=$(echo "$INTEGRAL_STR" | tr ',' '_')
    OUTDIR=$OUTBASE/$LABEL
    mkdir -p "$OUTDIR/logs" "$OUTDIR/work"

    echo "--- integral $((++n))/5: [$INTEGRAL_STR]  r=$r s=$s r+s=$rs ---"

    t0=$(date +%s)

    PYTHONUNBUFFERED=1 $PYTHON -u reduction/hierarchical_reduction.py \
        --topology         "$TOPOLOGY" \
        --integral         "$INTEGRAL_STR" \
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
        2>&1 | tee "$OUTDIR/logs/hierarchical.log" || true

    t1=$(date +%s)
    elapsed=$(( t1 - t0 ))
    mins=$(( elapsed / 60 ))
    secs=$(( elapsed % 60 ))

    # check success
    status="INCOMPLETE"
    if [[ -f "$OUTDIR/reduction.pkl" ]]; then
        status=$(python3 -c "
import pickle
r = pickle.load(open('$OUTDIR/reduction.pkl','rb'))
print('SUCCESS' if r.get('success') else 'FAILED')
" 2>/dev/null || echo "UNKNOWN")
    fi

    RESULTS+=("r+s=$rs  ${mins}m${secs}s  $status  [$INTEGRAL_STR]")
    echo "    => $status in ${mins}m${secs}s"
    echo ""

done < "$INTEGRAL_LIST"

echo "================================================================"
echo "BENCHMARK SUMMARY"
echo "================================================================"
for r in "${RESULTS[@]}"; do echo "  $r"; done
echo ""
echo "Rule of thumb for full run (700 integrals, 64 chunks, 47h):"
echo "  budget per integral ≈ 4.3 hours"
echo "================================================================"
