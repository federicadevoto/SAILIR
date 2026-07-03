#!/bin/bash
# Multi-node DDP training on Perlmutter.
#
# Run inside an interactive salloc on GPU nodes, wrapped in tmux so the
# allocation survives SSH disconnect:
#
#   tmux new -s sailir
#   salloc -N 4 -C gpu --gpus 16 -q interactive -t 04:00:00 -A m4539_g
#   bash training/nersc_perlmutter/allocation.sh
#   # Ctrl-B D to detach; `tmux a -t sailir` to re-attach later.
#
# Env switches:
#   TOPOLOGY=path      — topology dir rel. to repo root (default topology_input/pentagonbox).
#   SHARDS_DIR=path    — packed shards dir rel. to repo root (default data/pentagonbox_10x_packed).
#   SMOKE=1            — 5 epochs, smoke-sized shards. OUTPUT_DIR forced to
#                        checkpoints/<dataset>_smoke. Exercises rendezvous,
#                        NCCL, sharded loader, val pass, checkpoint save.
#                        Smoke shard counts must be multiples of world_size;
#                        adjust MAX_SMOKE_TRAIN/MAX_SMOKE_VAL if using fewer GPUs.
#   EPOCHS=N           — override epoch count (default 20).
#   BATCH_SIZE=N       — per-rank batch size (default 128). Effective batch
#                        is BATCH_SIZE × world_size. If you change this,
#                        scale --lr accordingly (linear or sqrt rule).
#   MAX_TRAIN_SHARDS=N — cap train shards (default: all). Useful for small
#                        end-to-end tests of the auto_resume + supervisor
#                        path without a 4-hour allocation.
#   N_VAL_SHARDS=N     — override val shard count (default 50).
#   OUTPUT_DIR=path    — checkpoint output dir (default checkpoints/<dataset>).
#                        Override to test against a non-default run dir.
#                        Must agree with the supervisor's OUTPUT_DIR when
#                        running under train_loop.sh.
#                        Each split needs n_shards >= world_size.
#   MODEL_VARIANT=name — which classifier class to train (default `nosubs`,
#                        i.e. IBPActionClassifierNoSubs). `full` selects
#                        IBPActionClassifier (subs encoder included, ~40%
#                        more params, same accuracy). Checkpoints from
#                        different variants are NOT interchangeable; use a
#                        distinct OUTPUT_DIR per variant.

set -euo pipefail
cd "$(dirname "$0")/../.."

module load pytorch/2.11.0

if [[ "${SLURM_JOB_ID:-}" == "" ]]; then
    echo "ERROR: not inside a SLURM allocation. Run salloc first." >&2
    exit 1
fi

# Rendezvous: first node in the allocation acts as master.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1

TOPOLOGY=${TOPOLOGY:-topology_input/pentagonbox}
SHARDS_DIR=${SHARDS_DIR:-data/pentagonbox_10x_packed}
SMOKE=${SMOKE:-0}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-128}
MODEL_VARIANT=${MODEL_VARIANT:-nosubs}
NUM_WORKERS=${NUM_WORKERS:-4}

# Derive a short dataset name from SHARDS_DIR for log/checkpoint naming.
DATASET=$(basename "$SHARDS_DIR" _packed)

if [[ "$SMOKE" == "1" ]]; then
    OUTPUT_DIR=checkpoints/${DATASET}_smoke
    LOG_TAG=smoke
    # Smoke counts must be multiples of world_size (16 for 4-node × 4-GPU).
    # If running on fewer GPUs (e.g. 1 node = 4 GPUs) lower these accordingly.
    MAX_SMOKE_TRAIN=${MAX_SMOKE_TRAIN:-64}
    MAX_SMOKE_VAL=${MAX_SMOKE_VAL:-16}
    EXTRA_ARGS=( --max_train_shards "$MAX_SMOKE_TRAIN" --n_val_shards "$MAX_SMOKE_VAL" --epochs 5 )
else
    OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/${DATASET}}
    LOG_TAG=full
    N_VAL_SHARDS=${N_VAL_SHARDS:-50}
    EXTRA_ARGS=( --n_val_shards "$N_VAL_SHARDS" --epochs "$EPOCHS" --auto_resume )
    [[ -n "${MAX_TRAIN_SHARDS:-}" ]] && EXTRA_ARGS+=( --max_train_shards "$MAX_TRAIN_SHARDS" )
fi
EXTRA_ARGS+=( --model_variant "$MODEL_VARIANT" )

mkdir -p "$OUTPUT_DIR" logs
LOG=logs/${DATASET}_${LOG_TAG}_$(date +%Y%m%d_%H%M%S).log

{
  echo "[$(date -Iseconds)] launch SMOKE=$SMOKE EPOCHS=$EPOCHS"
  echo "  SLURM_JOB_ID=$SLURM_JOB_ID  nodes=$SLURM_JOB_NUM_NODES"
  echo "  MASTER_ADDR=$MASTER_ADDR:$MASTER_PORT"
  echo "  OUTPUT_DIR=$OUTPUT_DIR"
} | tee -a "$LOG"

srun -l -u \
    --ntasks-per-node=4 \
    --gpus-per-task=1 \
    --cpus-per-task=32 \
    --gpu-bind=none \
    bash training/nersc_perlmutter/srun_task.sh \
        --topology         "$TOPOLOGY" \
        --shards_dir       "$SHARDS_DIR" \
        --buffer_shards    4 \
        --output_dir       "$OUTPUT_DIR" \
        --batch_size       "$BATCH_SIZE" \
        --lr               4e-4 \
        --prime            1009 \
        --device           cuda \
        --num_workers      "$NUM_WORKERS" \
        --checkpoint_every 1 \
        --log_every        50 \
        --seed             0 \
        "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$LOG"
