#!/bin/bash
# ============================================================================
# Hierarchical reduction launcher for a SHARED SINGLE MACHINE with no batch
# scheduler (e.g. tplx) — the --backend local counterpart of run_reduction.sh.
#
# Same recipe as run_reduction.sh (pentagonbox_nosym + --paper-masters-only,
# 1-CPU v7 workers fanned out wide, straggler escalation disabled, --resume
# for crash recovery) EXCEPT jobs are launched as direct subprocesses on THIS
# machine instead of submitted to Condor:
#     --backend local      (subprocess.Popen workers, no condor_submit/_q/_rm)
#     --max-cpus 120        (never use more than 120 cores at once -- this
#                             machine is shared; raise/lower for your box)
#
# To reduce a DIFFERENT integral: change INTEGRAL_STR and OUTDIR below.
# ============================================================================
set -e
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- change these two for your target ----
INTEGRAL_STR="1,1,1,1,1,1,1,1,-5,0,0"                 # 11-int target (8 props + 3 ISPs)
OUTDIR=$BASE/results/my_reduction                     # output dir
# ------------------------------------------

PYTHON=$BASE/venv/bin/python
MODEL=$BASE/checkpoints/pentagonbox_10x_loop_100/best_model.pt
TOPOLOGY=$BASE/topology_input/pentagonbox_nosym

mkdir -p $OUTDIR/logs $OUTDIR/work/logs $OUTDIR/work/results

set -x
PYTHONUNBUFFERED=1 $PYTHON -u $BASE/reduction/hierarchical_reduction.py \
    --topology $TOPOLOGY \
    --integral $INTEGRAL_STR \
    --output $OUTDIR/reduction.pkl \
    --work-dir $OUTDIR/work \
    --model-checkpoint $MODEL \
    --beam_width 40 --max_steps 1000000 --prime 1009 \
    --paper-masters-only \
    --use-v7-worker \
    --v7-cpus 1 \
    --backend local \
    --max-cpus 120 \
    --straggler-timeout 1000000000 \
    --straggler2-timeout 1000000000 \
    --check-interval 5 \
    --max-concurrent 1000 \
    --resume \
  > $OUTDIR/logs/hierarchical.log 2>&1 &
ORCH_PID=$!
set +x
echo "reduction orchestrator launched PID=$ORCH_PID"
echo "  log:     $OUTDIR/logs/hierarchical.log"
echo "  workdir: $OUTDIR/work"
echo "  result:  $OUTDIR/reduction.pkl"
echo
echo "Stop with: kill $ORCH_PID   (SIGTERM/SIGINT cleans up all running workers)"
