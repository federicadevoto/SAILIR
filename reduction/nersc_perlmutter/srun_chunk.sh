#!/bin/bash
# Per-chunk task wrapper. Launched once per srun task by allocation_cpu.sh.
# Mirrors training/nersc_perlmutter/srun_task.sh but for reduction.
#
# SLURM_PROCID  = this chunk's 0-based index (0 .. SLURM_NTASKS-1)
# SLURM_NTASKS  = total number of chunks

export CHUNK_ID=${SLURM_PROCID}
export N_CHUNKS=${SLURM_NTASKS}
export MAX_CPUS=${CPUS_PER_CHUNK:-8}

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/sailir:${REPO_ROOT}:${PYTHONPATH:-}"

# Per-chunk log so output from 64 tasks doesn't interleave in the job log.
LOG="${OUTBASE:-$REPO_ROOT/results/hexabox}/logs/chunk_${CHUNK_ID}.log"
mkdir -p "$(dirname "$LOG")"

exec bash "${REPO_ROOT}/reduction/run_reduction_chunk.sh" >> "$LOG" 2>&1
