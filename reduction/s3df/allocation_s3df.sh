#!/bin/bash
# Hexabox IBP reduction on SLAC S3DF batch farm.
#
# Submit with: sbatch reduction/s3df/allocation_s3df.sh
# (from repo root, on an sdfiana node)
#
# Uses the EPPTheory:QCD uninterruptible 256-core allocation.
# 32 chunks × 8 CPUs = 256 cores; 700 integrals / 32 chunks ≈ 22 per chunk.
# --resume skips already-finished integrals on resubmission.
#
# Env switches (all optional):
#   TOPOLOGY, INTEGRAL_LIST, MODEL, OUTBASE  — paths relative to repo root
#   CPUS_PER_CHUNK   — CPU budget per chunk (default 8; must match --cpus-per-task)
#   N_CHUNKS         — parallel chunks (default 32; must match --ntasks)
#SBATCH --job-name=hexabox_reduce
#SBATCH --account=EPPTheory:QCD
#SBATCH --partition=milano
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/reduce_%j.out
#SBATCH --error=logs/reduce_%j.err

set -uo pipefail
cd "$(dirname "$0")/../.."

# ── Python environment ────────────────────────────────────────────────────────
# Use the repo venv if it exists; otherwise fall back to whatever python is
# in PATH (e.g. after `module load` or inside a conda env).
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
N_CHUNKS=${N_CHUNKS:-32}

export PYTHONPATH="$(pwd)/sailir:$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUTBASE/logs" logs

{
  echo "[$(date -Iseconds)] Hexabox reduction on S3DF"
  echo "  SLURM_JOB_ID=$SLURM_JOB_ID"
  echo "  ntasks=$SLURM_NTASKS  cpus_per_task=$SLURM_CPUS_PER_TASK"
  echo "  chunks=$N_CHUNKS  cpus/chunk=$CPUS_PER_CHUNK"
  echo "  integral list: $INTEGRAL_LIST"
  echo "  model: $MODEL"
  echo "  output: $OUTBASE"
}

srun \
    --ntasks="$N_CHUNKS" \
    --cpus-per-task="$CPUS_PER_CHUNK" \
    bash reduction/nersc_perlmutter/srun_chunk.sh

echo "[$(date -Iseconds)] All chunks complete."
