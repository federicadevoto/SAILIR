#!/bin/bash
# ============================================================================
# Hierarchical reduction launcher for a SHARED SINGLE MACHINE with no batch
# scheduler (e.g. tplx) — the --backend local counterpart of run_reduction.sh.
#
# Reduces all integrals listed in INTEGRAL_LIST (one TB[...] per line)
# sequentially, using up to MAX_CPUS cores per reduction.
#
# Usage: bash reduction/run_reduction_local.sh   (from SAILIR repo root)
#        Run inside a screen session on tplx.
# ============================================================================
set -e
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- config ----------------------------------------------------------------
INTEGRAL_LIST=$BASE/reduction/integrals_to_reduce.txt   # one TB[...] per line
OUTBASE=$BASE/results/hexabox                           # one subdir per integral
MAX_CPUS=120                                            # total cores (shared machine)
# ----------------------------------------------------------------------------

PYTHON=$BASE/venv/bin/python
MODEL=$BASE/checkpoints/hexabox_100k/best_model.pt
TOPOLOGY=$BASE/topology_input/hexabox

echo "============================================================"
echo "Hexabox reduction run"
echo "  integrals: $INTEGRAL_LIST"
echo "  model:     $MODEL"
echo "  output:    $OUTBASE"
echo "  started:   $(date)"
echo "============================================================"

n=0
while IFS= read -r line || [[ -n "$line" ]]; do
    # skip blank lines and comments
    [[ -z "$line" || "$line" == \#* ]] && continue

    # parse TB[a1,...,a11] -> a1,...,a11
    INTEGRAL_STR=$(echo "$line" | sed 's/TB\[//;s/\]//')
    # make a safe directory name: replace commas with underscores
    LABEL=$(echo "$INTEGRAL_STR" | tr ',' '_')
    OUTDIR=$OUTBASE/$LABEL

    mkdir -p "$OUTDIR/logs" "$OUTDIR/work"

    echo ""
    echo "--- integral $((++n)): TB[$INTEGRAL_STR] ---"
    echo "    output: $OUTDIR"

    PYTHONUNBUFFERED=1 $PYTHON -u $BASE/reduction/hierarchical_reduction.py \
        --topology    "$TOPOLOGY" \
        --integral    "$INTEGRAL_STR" \
        --output      "$OUTDIR/reduction.pkl" \
        --work-dir    "$OUTDIR/work" \
        --model-checkpoint "$MODEL" \
        --beam_width 40 --max_steps 1000000 --prime 1009 \
        --no-paper-masters-only \
        --use-v7-worker \
        --v7-cpus 1 \
        --backend local \
        --max-cpus $MAX_CPUS \
        --straggler-timeout 1000000000 \
        --straggler2-timeout 1000000000 \
        --check-interval 5 \
        --max-concurrent 1000 \
        --resume \
        2>&1 | tee "$OUTDIR/logs/hierarchical.log"

    echo "    done: $(date)"
done < "$INTEGRAL_LIST"

echo ""
echo "============================================================"
echo "All integrals done: $(date)"
echo "Results in: $OUTBASE"
echo "============================================================"
