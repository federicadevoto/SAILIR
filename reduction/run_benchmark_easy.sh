#!/bin/bash
# Times 10 easy hexabox integrals sequentially on S3DF.
# Run from repo root in a screen session on sdfiana:
#   screen -S bench
#   bash reduction/run_benchmark_easy.sh

set -uo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE"

if [[ -f venv/bin/activate ]]; then source venv/bin/activate; fi

TOPOLOGY=${TOPOLOGY:-topology_input/hexabox}
MODEL=${MODEL:-checkpoints/hexabox_100k/best_model.pt}
OUTBASE=${OUTBASE:-results/hexabox_bench_easy}
INTEGRAL_LIST=${INTEGRAL_LIST:-reduction/benchmark_hexabox_easy.txt}
MAX_CPUS=${MAX_CPUS:-8}
PYTHON=${PYTHON:-python}

export PYTHONPATH="$BASE/sailir:$BASE:${PYTHONPATH:-}"
mkdir -p "$OUTBASE"

echo "========================================================"
echo "Hexabox easy benchmark  $(date)"
echo "  model:    $MODEL"
echo "  MAX_CPUS: $MAX_CPUS"
echo "========================================================"

n=0
while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    (( ++n ))

    LABEL=$(echo "$line" | tr ',' '_')
    OUTDIR="$OUTBASE/$LABEL"
    mkdir -p "$OUTDIR/logs" "$OUTDIR/work"

    # compute r+s for display
    rs=$(python3 -c "
idx=[int(x) for x in '$line'.split(',')]
print(sum(x for x in idx if x>0)+sum(-x for x in idx if x<0))
" 2>/dev/null || echo "?")

    echo ""
    echo "--- integral $n/10  r+s=$rs  [$line] ---"
    t0=$(date +%s)

    PYTHONUNBUFFERED=1 $PYTHON -u "$BASE/reduction/hierarchical_reduction.py" \
        --topology         "$TOPOLOGY" \
        --integral="$line" \
        --output           "$OUTDIR/reduction.pkl" \
        --work-dir         "$OUTDIR/work" \
        --model-checkpoint "$MODEL" \
        --beam_width 40 --max_steps 1000000 --prime 1009 \
        --no-paper-masters-only \
        --use-v7-worker \
        --v7-cpus 1 \
        --backend local \
        --max-cpus "$MAX_CPUS" \
        --straggler-timeout 1000000000 \
        --straggler2-timeout 1000000000 \
        --check-interval 5 \
        --max-concurrent 1000 \
        2>&1 | tee "$OUTDIR/logs/hierarchical.log" || true

    t1=$(date +%s)
    elapsed=$(( t1 - t0 ))
    mins=$(( elapsed / 60 ))
    secs=$(( elapsed % 60 ))

    status="INCOMPLETE"
    if [[ -f "$OUTDIR/reduction.pkl" ]]; then
        status=$(python3 -c "
import pickle
r=pickle.load(open('$OUTDIR/reduction.pkl','rb'))
print('SUCCESS' if r.get('success') else 'FAILED')
" 2>/dev/null || echo "UNKNOWN")
    fi

    echo "  => $status in ${mins}m${secs}s"

done < "$INTEGRAL_LIST"

echo ""
echo "========================================================"
echo "DONE  $(date)"
echo "========================================================"
