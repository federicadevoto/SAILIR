#!/bin/bash
# Hexabox IBP reduction on S3DF — SLURM array job.
#
# One task per integral, 32 CPUs per task.
# %8 limits concurrent tasks to 8 → 8×32 = 256 cores (full priority allocation).
#
# Submit from repo root:
#   mkdir -p logs
#   sbatch reduction/s3df/array_job.sh
#
# Resubmit to retry timed-out or failed integrals (skip-if-done is automatic):
#   sbatch reduction/s3df/array_job.sh
#
# Check progress:
#   find results/hexabox -name "reduction.pkl" | wc -l     # succeeded
#   find results/hexabox -name "reduction.timeout" | wc -l # timed out
#
# Env switches (all optional):
#   INTEGRAL_LIST, MODEL, TOPOLOGY, OUTBASE — paths relative to repo root
#   INTEGRAL_TIMEOUT — seconds per integral (default 10500 = 2h55m)
#SBATCH --job-name=hb_reduce
#SBATCH --account=epptheory:qcd
#SBATCH --partition=milano
#SBATCH --qos=normal
#SBATCH --array=0-699
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=logs/reduce_array_%A_%a.out
#SBATCH --error=logs/reduce_array_%A_%a.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

if [[ -f venv/bin/activate ]]; then
    source venv/bin/activate
fi

INTEGRAL_LIST=${INTEGRAL_LIST:-reduction/integrals_to_reduce.txt}
MODEL=${MODEL:-checkpoints/hexabox_13M_biased/best_model.pt}
TOPOLOGY=${TOPOLOGY:-topology_input/hexabox}
OUTBASE=${OUTBASE:-results/hexabox}
INTEGRAL_TIMEOUT=${INTEGRAL_TIMEOUT:-10500}   # 5 min buffer before 3h SLURM limit

# Read integral for this task (0-indexed, skip blank lines and comments)
INTEGRAL=$(awk 'NF && !/^#/ { if (i++ == n) { print; exit } }' \
           n="$SLURM_ARRAY_TASK_ID" "$INTEGRAL_LIST")

if [[ -z "$INTEGRAL" ]]; then
    echo "No integral at index $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

LABEL=$(echo "$INTEGRAL" | tr ',' '_')
OUTDIR="$OUTBASE/$LABEL"

if [[ -f "$OUTDIR/reduction.pkl" || -f "$OUTDIR/reduction.timeout" ]]; then
    echo "[task $SLURM_ARRAY_TASK_ID] skipping $INTEGRAL (already done)"
    exit 0
fi

mkdir -p "$OUTDIR/logs" "$OUTDIR/work"
echo "[task $SLURM_ARRAY_TASK_ID] reducing $INTEGRAL  (array $SLURM_ARRAY_JOB_ID)"

PYTHONUNBUFFERED=1 timeout "$INTEGRAL_TIMEOUT" \
    python -u reduction/hierarchical_reduction.py \
    --topology         "$TOPOLOGY" \
    --integral="$INTEGRAL" \
    --output           "$OUTDIR/reduction.pkl" \
    --work-dir         "$OUTDIR/work" \
    --model-checkpoint "$MODEL" \
    --beam_width 40 --max_steps 1000000 --prime 1009 \
    --no-paper-masters-only \
    --use-v7-worker \
    --v7-cpus 1 \
    --backend local \
    --max-cpus 32 \
    --straggler-timeout 1000000000 \
    --straggler2-timeout 1000000000 \
    --check-interval 5 \
    --max-concurrent 1000 \
    --resume \
    2>&1 | tee "$OUTDIR/logs/hierarchical.log"
PIPE_RC=${PIPESTATUS[0]}

if [[ $PIPE_RC -eq 124 ]]; then
    echo "[task $SLURM_ARRAY_TASK_ID] TIMEOUT on $INTEGRAL"
    touch "$OUTDIR/reduction.timeout"
elif [[ $PIPE_RC -ne 0 && ! -f "$OUTDIR/reduction.pkl" ]]; then
    echo "[task $SLURM_ARRAY_TASK_ID] FAILED (rc=$PIPE_RC) on $INTEGRAL"
fi
