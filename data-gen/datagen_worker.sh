#!/bin/bash
# ============================================================================
# SAILIR data-gen worker (Stage 1) — generates ONE shard of raw training data
# for the topology in $TOPOLOGY. Invoked by Condor (one process per worker) via
# the JDL that submit_datagen.sh writes; NOT run by hand. Each worker writes one
# JSONL file using a disjoint block of random seeds.
#
#   args:  <worker_id> [n_scrambles] [output_dir]
#   env:   SAILIR_DIR   repo root
#          TOPOLOGY     topology dir relative to SAILIR_DIR (must hold IBP, LI, masters)
#          DATASET      dataset name (used for the default output dir)
#          PYTHON       python interpreter (with torch + the sailir package)
#          SEED_STRIDE  per-worker seed block size (default 1000000, >> n_scrambles)
#          SEED_OFFSET  global seed offset (default 0; bump to make a disjoint dataset)
# ============================================================================
set -e
WORKER_ID=$1
N_SCRAMBLES=${2:-1000}
: "${SAILIR_DIR:?set SAILIR_DIR to the repo root}"
: "${TOPOLOGY:?set TOPOLOGY (e.g. topology_input/pentagonbox)}"
OUTPUT_DIR=${3:-${SAILIR_DIR}/data/${DATASET:?set DATASET}_raw_jsonl}
PYTHON=${PYTHON:-python3}
SEED_STRIDE=${SEED_STRIDE:-1000000}
SEED_OFFSET=${SEED_OFFSET:-0}

cd "${SAILIR_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Disjoint seed range per worker: stride >> n_scrambles so seeds never overlap.
START_SEED=$((WORKER_ID * SEED_STRIDE + SEED_OFFSET))
OUTPUT_FILE="${OUTPUT_DIR}/multisector_data_worker${WORKER_ID}.jsonl"

echo "=== data-gen worker ${WORKER_ID}: topology=${TOPOLOGY}"
echo "    seeds=[${START_SEED}, +${N_SCRAMBLES})  ->  ${OUTPUT_FILE}"

PYTHONUNBUFFERED=1 "${PYTHON}" -u data-gen/generate_multisector_data.py \
    --topology    "${SAILIR_DIR}/${TOPOLOGY}" \
    --ibp_path    "${SAILIR_DIR}/${TOPOLOGY}/IBP" \
    --li_path     "${SAILIR_DIR}/${TOPOLOGY}/LI" \
    --n_scrambles ${N_SCRAMBLES} \
    --start_seed  ${START_SEED} \
    --output      "${OUTPUT_FILE}" \
    --prime 1009 --min_steps 5 --max_steps 25 \
    --bias-low-s-elim --bias-dots-elim

echo "worker ${WORKER_ID} done."
