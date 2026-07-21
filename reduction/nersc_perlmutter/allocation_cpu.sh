#!/bin/bash
# Hexabox IBP reduction on Perlmutter CPU nodes.
#
# Submits via: sbatch reduction/nersc_perlmutter/allocation_cpu.sh
# Or runs inside a manual salloc (interactive queue):
#   salloc -N 4 -C cpu -q interactive -t 04:00:00 -A m4539
#   bash reduction/nersc_perlmutter/allocation_cpu.sh
#
# Parallelization:
#   4 nodes × 16 tasks/node × 8 CPUs/task = 64 simultaneous reductions, 512 cores.
#   700 integrals / 64 chunks ≈ 11 integrals per chunk.
#   Each chunk runs hierarchical_reduction.py with MAX_CPUS=8 (8 concurrent workers).
#   --resume skips already-finished integrals on restart.
#
# Env switches:
#   TOPOLOGY=path        topology dir (default topology_input/hexabox)
#   INTEGRAL_LIST=path   one TB[...] per line (default reduction/integrals_to_reduce.txt)
#   MODEL=path           model checkpoint (default checkpoints/hexabox_100k/best_model.pt)
#   OUTBASE=path         results root (default results/hexabox)
#   CPUS_PER_CHUNK=N     CPU budget per chunk; must match --cpus-per-task (default 8)
#   N_TASKS_PER_NODE=N   parallel chunks per node (default 16; 16×8=128=full node)
#SBATCH --job-name=hexabox_reduce
#SBATCH -A m4539
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH --time=47:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/reduce_%j.out
#SBATCH --error=logs/reduce_%j.err

set -uo pipefail
cd "$(dirname "$0")/../.."

module load pytorch/2.11.0

if [[ "${SLURM_JOB_ID:-}" == "" ]]; then
    echo "ERROR: not inside a SLURM allocation. Run salloc or sbatch." >&2
    exit 1
fi

export TOPOLOGY=${TOPOLOGY:-topology_input/hexabox}
export INTEGRAL_LIST=${INTEGRAL_LIST:-reduction/integrals_to_reduce.txt}
export MODEL=${MODEL:-checkpoints/hexabox_100k/best_model.pt}
export OUTBASE=${OUTBASE:-results/hexabox}
export CPUS_PER_CHUNK=${CPUS_PER_CHUNK:-8}
N_TASKS_PER_NODE=${N_TASKS_PER_NODE:-16}

mkdir -p "$OUTBASE/logs" logs

{
  echo "[$(date -Iseconds)] Hexabox CPU reduction"
  echo "  SLURM_JOB_ID=$SLURM_JOB_ID  nodes=$SLURM_JOB_NUM_NODES"
  echo "  total tasks: $SLURM_NTASKS  (${N_TASKS_PER_NODE}/node × ${CPUS_PER_CHUNK} CPUs/task)"
  echo "  integral list: $INTEGRAL_LIST"
  echo "  model: $MODEL"
  echo "  output: $OUTBASE"
}

srun \
    --ntasks-per-node="$N_TASKS_PER_NODE" \
    --cpus-per-task="$CPUS_PER_CHUNK" \
    bash reduction/nersc_perlmutter/srun_chunk.sh

echo "[$(date -Iseconds)] All chunks complete."
