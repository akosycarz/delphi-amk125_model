#!/bin/bash
#PBS -N delphi_sensitivity
#PBS -q v1_medium72
#PBS -l select=1:ncpus=32:mem=128gb
#PBS -l walltime=72:00:00
#PBS -j oe

set -euo pipefail

# ---------------------------------------------------------------------------
# EDIT THESE PATHS BEFORE SUBMITTING
# ---------------------------------------------------------------------------
PROJECT_ROOT="${HOME}/delphi-amk125_model"
PROJECT_DIR="${PROJECT_ROOT}/Delphi"
SENSITIVITY_ROOT="${PROJECT_ROOT}/data_sensitive_analysis"
MODELS_ROOT="${PROJECT_DIR}/models"
RESULTS_ROOT="${PROJECT_DIR}/results/sensitivity_complete_case"
# ---------------------------------------------------------------------------

MODEL_NAMES=(
  "ukb_amk125_clinical_icd"
  "ukb_amk125_clinical_demographics_icd"
  "ukb_amk125_clinical_demographics_ukb_icd"
  "ukb_amk125_clinical_demographics_ukb_biochem_icd"
  "ukb_amk125_clinical_demographics_ukb_biochem_icd_self_reported"
)

TASK_ID="${MODEL_INDEX:-${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-}}}"
if [[ -z "${TASK_ID}" ]]; then
  echo "ERROR: MODEL_INDEX is unset. Submit with qsub -v MODEL_INDEX=0..4." >&2
  exit 1
fi
if (( TASK_ID < 0 || TASK_ID >= ${#MODEL_NAMES[@]} )); then
  echo "ERROR: Invalid PBS array index: ${TASK_ID}" >&2
  exit 1
fi

MODEL_NAME="${MODEL_NAMES[$TASK_ID]}"
INPUT_PATH="${SENSITIVITY_ROOT}/${MODEL_NAME}"
CHECKPOINT="${MODELS_ROOT}/${MODEL_NAME}/ckpt.pt"
OUTPUT_PATH="${RESULTS_ROOT}/${MODEL_NAME}"

for required_file in \
  "${PROJECT_DIR}/run_evaluation.py" \
  "${PROJECT_DIR}/evaluate_pr_original_cohorts.py" \
  "${PROJECT_DIR}/sample_checkpoint_trajectories.py" \
  "${PROJECT_DIR}/evaluate_sex_and_gaps_fast.py" \
  "${INPUT_PATH}/test.bin" \
  "${INPUT_PATH}/token_dictionary.csv" \
  "${CHECKPOINT}"; do
  if [[ ! -s "${required_file}" ]]; then
    echo "ERROR: Required file is missing or empty: ${required_file}" >&2
    exit 1
  fi
done

eval "$(~/anaconda3/bin/conda shell.bash hook)"
conda activate delphi

export OMP_NUM_THREADS="${PBS_NCPUS:-32}"
export MKL_NUM_THREADS="${PBS_NCPUS:-32}"
export OPENBLAS_NUM_THREADS="${PBS_NCPUS:-32}"

mkdir -p \
  "${OUTPUT_PATH}/main_evaluation" \
  "${OUTPUT_PATH}/pr_original_cohorts" \
  "${OUTPUT_PATH}/trajectories" \
  "${OUTPUT_PATH}/sex_and_gaps"
cd "${PROJECT_DIR}"

echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Array index: ${TASK_ID}"
echo "Model: ${MODEL_NAME}"
echo "Sensitivity data: ${INPUT_PATH}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Results: ${OUTPUT_PATH}"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-not_set}"

python -u run_evaluation.py \
  --input_path "${INPUT_PATH}" \
  --model_ckpt_path "${CHECKPOINT}" \
  --output_path "${OUTPUT_PATH}/main_evaluation" \
  --model_name "${MODEL_NAME}_complete_case_sensitivity" \
  --split test \
  --device cpu \
  --batch_size 64 \
  --disease_chunk_size 64 \
  --dataset_subset_size -1 \
  --count_threshold 100 \
  --no-extended_analysis

echo "Running original-cohort precision-recall analysis: $(date)"
python -u evaluate_pr_original_cohorts.py \
  --input-path "${INPUT_PATH}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_PATH}/pr_original_cohorts" \
  --model-name "${MODEL_NAME}_complete_case_sensitivity" \
  --split test \
  --block-size 0 \
  --device cpu \
  --batch-size 64 \
  --disease-chunk-size 16 \
  --dataset-subset-size -1 \
  --filter-min-total 100

echo "Sampling checkpoint trajectories: $(date)"
python -u sample_checkpoint_trajectories.py \
  --checkpoint "${CHECKPOINT}" \
  --data-dir "${INPUT_PATH}" \
  --output-dir "${OUTPUT_PATH}/trajectories" \
  --split test \
  --device cpu \
  --n-patients 500 \
  --samples-per-patient 1 \
  --cutoff-age 60 \
  --max-age 85 \
  --max-new-tokens 128 \
  --seed 1337

echo "Running sex and prediction-gap analysis: $(date)"
python -u evaluate_sex_and_gaps_fast.py \
  --checkpoint "${CHECKPOINT}" \
  --data-dir "${INPUT_PATH}" \
  --output-dir "${OUTPUT_PATH}/sex_and_gaps" \
  --model-name "${MODEL_NAME}_complete_case_sensitivity" \
  --gaps-months 0 6 12 60 120 \
  --split test \
  --device cpu \
  --batch-size 32 \
  --disease-chunk-size 64 \
  --dataset-subset-size -1 \
  --min-events 100 \
  --seed 1337

echo "Completed ${MODEL_NAME}: $(date)"
