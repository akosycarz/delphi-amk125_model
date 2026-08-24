#!/bin/bash
#PBS -N delphi_fast_gap
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o /rds/general/user/amk125/ephemeral/logs
#PBS -e /rds/general/user/amk125/ephemeral/logs
# Fast AUC-only sex and prediction-gap evaluation.
# Model inference is performed once and reused across all requested gaps.
#
# Submit one model:
#   qsub -v MODEL_NAME=clinical_icd evaluate_sex_and_gaps_fast_select_model.sh
#
# Custom gaps (colon-separated because PBS uses commas between variables):
#   qsub -v MODEL_NAME=clinical_icd,GAPS_MONTHS=0:6:12:60:120 \
#     evaluate_sex_and_gaps_fast_select_model.pbs
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/rds/general/user/amk125/home/delphi-amk125_model}"
CODE_DIR="${CODE_DIR:-${PROJECT_ROOT}/Delphi}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
MODELS_ROOT="${MODELS_ROOT:-${CODE_DIR}/models}"
RESULTS_ROOT="${RESULTS_ROOT:-${CODE_DIR}/results}"
PYTHON="${PYTHON:-/rds/general/user/amk125/home/anaconda3/envs/delphi/bin/python}"
LOG_DIR="${LOG_DIR:-/rds/general/user/amk125/ephemeral/logs}"
MODEL_NAME="${MODEL_NAME:-}"
DATA_NAME="${DATA_NAME:-${MODEL_NAME}}"
MODEL_LABEL="${MODEL_LABEL:-${MODEL_NAME}}"
GAPS_MONTHS="${GAPS_MONTHS:-0:6:12:60:120}"
SPLIT="${SPLIT:-test}"
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DISEASE_CHUNK_SIZE="${DISEASE_CHUNK_SIZE:-64}"
NO_EVENT_TOKEN_RATE="${NO_EVENT_TOKEN_RATE:-5}"
DATASET_SUBSET_SIZE="${DATASET_SUBSET_SIZE:--1}"
COUNT_THRESHOLD="${COUNT_THRESHOLD:-100}"
FORCE="${FORCE:-0}"
fail() {
  echo "ERROR: $*" >&2
  exit 2
}
[[ -n "${MODEL_NAME}" ]] || fail \
  "MODEL_NAME is required; use qsub -v MODEL_NAME=clinical_icd ..."
CHECKPOINT="${CHECKPOINT:-${MODELS_ROOT}/${MODEL_NAME}/ckpt.pt}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}/${DATA_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/${MODEL_NAME}/fast_sex_gap_evaluation}"
SCRIPT="${CODE_DIR}/evaluate_sex_and_gaps_fast.py"
JOB_ID="${PBS_JOBID:-manual}"
LOG_FILE="${LOG_DIR}/fast_sex_gap_${MODEL_NAME}_${JOB_ID}.log"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
exec >"${LOG_FILE}" 2>&1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${PBS_NCPUS:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
IFS=':' read -r -a GAP_ARRAY <<< "${GAPS_MONTHS}"
[[ "${#GAP_ARRAY[@]}" -gt 0 ]] || fail "GAPS_MONTHS is empty"
echo "Started:        $(date)"
echo "Host:           $(hostname)"
echo "PBS job:        ${JOB_ID}"
echo "Model:          ${MODEL_NAME}"
echo "Model label:    ${MODEL_LABEL}"
echo "Checkpoint:     ${CHECKPOINT}"
echo "Dataset:        ${DATA_DIR}"
echo "Output:         ${OUTPUT_DIR}"
echo "Gaps (months):  ${GAP_ARRAY[*]}"
echo "Device:         ${DEVICE}"
echo "Min events:     ${COUNT_THRESHOLD}"
echo "Log:            ${LOG_FILE}"
[[ -x "${PYTHON}" ]] || fail "Python not found: ${PYTHON}"
[[ -f "${SCRIPT}" ]] || fail "Fast evaluator not found: ${SCRIPT}"
[[ -f "${CHECKPOINT}" ]] || fail "Checkpoint not found: ${CHECKPOINT}"
[[ -f "${DATA_DIR}/${SPLIT}.bin" ]] || fail \
  "Data split not found: ${DATA_DIR}/${SPLIT}.bin"
[[ -f "${DATA_DIR}/token_dictionary.csv" ]] || fail \
  "Token dictionary not found: ${DATA_DIR}/token_dictionary.csv"
if [[ "${FORCE}" != "1" && -s "${OUTPUT_DIR}/auc_results.csv" ]]; then
  echo "Existing result found; use FORCE=1 to rerun."
  exit 0
fi
cd "${CODE_DIR}"
"${PYTHON}" "${SCRIPT}" \
  --checkpoint "${CHECKPOINT}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-name "${MODEL_LABEL}" \
  --gaps-months "${GAP_ARRAY[@]}" \
  --split "${SPLIT}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --disease-chunk-size "${DISEASE_CHUNK_SIZE}" \
  --no-event-token-rate "${NO_EVENT_TOKEN_RATE}" \
  --dataset-subset-size "${DATASET_SUBSET_SIZE}" \
  --min-events "${COUNT_THRESHOLD}"
for REQUIRED in auc_results.csv auc_results_all_gaps.csv \
  auc_results_all_gaps_filtered.csv \
  evaluation_coverage_by_gap_and_sex.csv; do
  [[ -s "${OUTPUT_DIR}/${REQUIRED}" ]] || fail \
    "Fast evaluation did not create ${OUTPUT_DIR}/${REQUIRED}"
done
touch "${OUTPUT_DIR}/FAST_SEX_GAP_EVALUATION_COMPLETE"
echo "Completed: $(date)"
echo "Results:   ${OUTPUT_DIR}"