#!/bin/bash
# Per-rank task wrapper for Perlmutter multi-node DDP training.
#
# srun launches this script once per GPU (--ntasks-per-node=4 over N nodes).
# We translate SLURM's per-task env vars into the names train_classifier.py
# expects (which match torchrun's convention) and exec the trainer.
#
# Not intended to be run by hand — use allocation.sh.

set -euo pipefail

export RANK=${SLURM_PROCID}
export LOCAL_RANK=${SLURM_LOCALID}
export WORLD_SIZE=${SLURM_NTASKS}

# sailir/ package lives at repo root — add it to the path so train_classifier.py
# can import classifier, topology, etc. directly.
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/sailir:${REPO_ROOT}:${PYTHONPATH:-}"

exec python -u training/train_classifier.py "$@"
