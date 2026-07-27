#!/bin/bash
# Hexabox IBP reduction on SLAC S3DF batch farm.
#
# Submit with: sbatch reduction/s3df/allocation_s3df.sh
# (from repo root, on an sdfiana node)
#
# 16 chunks × 8 CPUs = 128 cores running as background processes on one node.
# 700 integrals / 16 chunks ≈ 44 per chunk.
# --resume skips already-finished integrals on resubmission.
#
# Env switches (all optional):
#   TOPOLOGY, INTEGRAL_LIST, MODEL, OUTBASE  — paths relative to repo root
#   CPUS_PER_CHUNK   — CPU budget per chunk (default 8)
#   N_CHUNKS         — parallel chunks (default 16)
#SBATCH --job-name=hexabox_reduce
#SBATCH --account=epptheory:qcd
#SBATCH --partition=milano
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --output=logs/reduce_%j.out
#SBATCH --error=logs/reduce_%j.err

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

if [[ -f venv/bin/activate ]]; then
    source venv/bin/activate
fi

if [[ "${SLURM_JOB_ID:-}" == "" ]]; then
    echo "ERROR: not inside a SLURM allocation. Use sbatch." >&2
    exit 1
fi

export TOPOLOGY=${TOPOLOGY:-topology_input/hexabox}
export INTEGRAL_LIST=${INTEGRAL_LIST:-reduction/integrals_to_reduce.txt}
export MODEL=${MODEL:-checkpoints/hexabox_100k/best_model.pt}
export OUTBASE=${OUTBASE:-results/hexabox}
export CPUS_PER_CHUNK=${CPUS_PER_CHUNK:-8}
export N_CHUNKS=${N_CHUNKS:-16}

export PYTHONPATH="$(pwd)/sailir:$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUTBASE/logs" logs

{
  echo "[$(date -Iseconds)] Hexabox reduction on S3DF"
  echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
  echo "  chunks=$N_CHUNKS  cpus/chunk=$CPUS_PER_CHUNK"
  echo "  integral list: $INTEGRAL_LIST"
  echo "  model: $MODEL"
  echo "  output: $OUTBASE"
}

# Launch all chunks as background processes on this node.
for i in $(seq 0 $(( N_CHUNKS - 1 ))); do
    CHUNK_ID=$i MAX_CPUS=$CPUS_PER_CHUNK \
        bash reduction/run_reduction_chunk.sh \
        >> "$OUTBASE/logs/chunk_${i}.log" 2>&1 &
done

echo "[$(date -Iseconds)] Launched $N_CHUNKS chunks, waiting..."
wait
echo "[$(date -Iseconds)] All chunks complete."
