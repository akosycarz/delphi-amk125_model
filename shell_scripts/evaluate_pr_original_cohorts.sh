#!/bin/bash
#PBS -N delphi_pr_original
#PBS -J 1-5
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=12:00:00
#PBS -o /rds/general/user/amk125/ephemeral/logs
#PBS -e /rds/general/user/amk125/ephemeral/logs

# Submit all models:
#   qsub shell_scripts/evaluate_pr_original_cohorts.sh
# Submit one model, for example Model 1:
#   qsub -J 1 shell_scripts/evaluate_pr_original_cohorts.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/rds/general/user/amk125/home/delphi-amk125_model}"
PROJECT_DIR="${PROJECT_DIR:-${PROJECT_ROOT}/Delphi}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
MODELS_ROOT="${MODELS_ROOT:-${PROJECT_DIR}/models}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/results_original_pr}"
PYTHON="${PYTHON:-/rds/general/user/amk125/home/anaconda3/envs/delphi/bin/python}"
LOG_DIR="${LOG_DIR:-/rds/general/user/amk125/ephemeral/logs}"

SPLIT="${SPLIT:-val}"
BLOCK_SIZE="${BLOCK_SIZE:-80}"
OFFSET_DAYS="${OFFSET_DAYS:-0.1}"
FILTER_MIN_TOTAL="${FILTER_MIN_TOTAL:-100}"
MIN_BAND_CASES="${MIN_BAND_CASES:-2}"
MIN_BAND_CONTROLS="${MIN_BAND_CONTROLS:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DISEASE_CHUNK_SIZE="${DISEASE_CHUNK_SIZE:-16}"
DEVICE="${DEVICE:-cpu}"
NO_EVENT_TOKEN_RATE="${NO_EVENT_TOKEN_RATE:-5}"

DATASETS=(
  ukb_amk125_clinical_icd
  ukb_amk125_clinical_demographics_icd
  ukb_amk125_clinical_demographics_ukb_icd
  ukb_amk125_clinical_demographics_ukb_biochem_icd
  ukb_amk125_clinical_demographics_ukb_biochem_icd_self_reported
)

OUTPUT_NAMES=(
  clinical_icd
  clinical_demographics_icd
  clinical_demographics_ukb_icd
  clinical_demographics_ukb_biochem_icd
  clinical_demographics_ukb_biochem_icd_self_reported
)

MODEL_LABELS=("Model 1" "Model 2" "Model 3" "Model 4" "Model 5")

ICD_CODES=(
  I10 Z86 Z92 E78 K57 Z87 Z88 Z85 K44 I25 J45 E11 Z90 Z51 I48
)

ARRAY_INDEX="${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-}}"
if [[ -z "${ARRAY_INDEX}" || "${ARRAY_INDEX}" -lt 1 || "${ARRAY_INDEX}" -gt 5 ]]; then
  echo "ERROR: PBS array index must be 1-5; got '${ARRAY_INDEX}'" >&2
  exit 2
fi

I=$((ARRAY_INDEX - 1))
DATASET="${DATASETS[$I]}"
OUTPUT_NAME="${OUTPUT_NAMES[$I]}"
MODEL_LABEL="${MODEL_LABELS[$I]}"
DATA_DIR="${DATA_ROOT}/${DATASET}"
CHECKPOINT="${MODELS_ROOT}/${DATASET}/ckpt.pt"
OUTPUT_DIR="${RESULTS_ROOT}/${OUTPUT_NAME}/evaluation"
SCRIPT="${PROJECT_DIR}/evaluate_pr_original_cohorts.py"
JOB_ID="${PBS_JOBID:-manual}"
MPL_CACHE="${TMPDIR:-/tmp}/matplotlib-${JOB_ID}-${ARRAY_INDEX}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${MPL_CACHE}"
export MPLCONFIGDIR="${MPL_CACHE}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${PBS_NCPUS:-${NCPUS:-4}}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

[[ -x "${PYTHON}" ]] || fail "Python is not executable: ${PYTHON}"
[[ -f "${SCRIPT}" ]] || fail "Evaluator not found: ${SCRIPT}"
[[ -f "${DATA_DIR}/${SPLIT}.bin" ]] || fail "Input split not found: ${DATA_DIR}/${SPLIT}.bin"
[[ -f "${DATA_DIR}/token_dictionary.csv" ]] || fail "Dictionary not found: ${DATA_DIR}/token_dictionary.csv"
[[ -f "${CHECKPOINT}" ]] || fail "Checkpoint not found: ${CHECKPOINT}"

echo "Started:        $(date)"
echo "Host:           $(hostname)"
echo "PBS job:        ${JOB_ID}"
echo "Array index:    ${ARRAY_INDEX}"
echo "Model:          ${MODEL_LABEL}"
echo "Dataset:        ${DATA_DIR}"
echo "Checkpoint:     ${CHECKPOINT}"
echo "Split:          ${SPLIT}"
echo "Output:         ${OUTPUT_DIR}"
echo "Device:         ${DEVICE}"

cd "${PROJECT_DIR}"
"${PYTHON}" "${SCRIPT}" \
  --input-path "${DATA_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-name "${MODEL_LABEL}" \
  --split "${SPLIT}" \
  --block-size "${BLOCK_SIZE}" \
  --offset-days "${OFFSET_DAYS}" \
  --age-start 40 \
  --age-stop 80 \
  --age-step 5 \
  --filter-min-total "${FILTER_MIN_TOTAL}" \
  --selected-icd-codes "${ICD_CODES[@]}" \
  --min-band-cases "${MIN_BAND_CASES}" \
  --min-band-controls "${MIN_BAND_CONTROLS}" \
  --no-event-token-rate "${NO_EVENT_TOKEN_RATE}" \
  --batch-size "${BATCH_SIZE}" \
  --disease-chunk-size "${DISEASE_CHUNK_SIZE}" \
  --device "${DEVICE}" \
  --save-predictions

for REQUIRED in pr_curves.csv pr_summary.csv pr_predictions.csv pr_run_config.csv; do
  [[ -s "${OUTPUT_DIR}/${REQUIRED}" ]] || fail "Missing output: ${OUTPUT_DIR}/${REQUIRED}"
done

touch "${OUTPUT_DIR}/PR_EVALUATION_COMPLETE"
echo "Completed:      $(date)"
echo "Results:        ${OUTPUT_DIR}"
