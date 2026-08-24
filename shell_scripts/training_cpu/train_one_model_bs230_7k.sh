#!/bin/bash
#PBS -N train_bs230_7k
#PBS -l select=1:ncpus=32:mem=540gb
#PBS -l walltime=72:00:00
# Keep stdout and stderr as separate, array-task-specific files. The trailing
# slash tells PBS to place its uniquely named job files inside this directory.
#PBS -j n
#PBS -o /rds/general/user/amk125/ephemeral/logs/
#PBS -e /rds/general/user/amk125/ephemeral/logs/

set -euo pipefail

# Allow PyTorch, NumPy, MKL/OpenBLAS and NumExpr to use every CPU allocated by
# PBS. PBS_NCPUS is supplied by PBS Pro; use 32 only as a safe fallback.
NCPUS=${PBS_NCPUS:-32}
export OMP_NUM_THREADS=${NCPUS}
export MKL_NUM_THREADS=${NCPUS}
export OPENBLAS_NUM_THREADS=${NCPUS}
export NUMEXPR_NUM_THREADS=${NCPUS}
export VECLIB_MAXIMUM_THREADS=${NCPUS}

PROJECT_DIR="/rds/general/user/amk125/home/delphi-amk125_model/Delphi"
DATA_ROOT="/rds/general/user/amk125/home/delphi-amk125_model/data"
PYTHON="/rds/general/user/amk125/home/anaconda3/envs/delphi/bin/python"

BLOCK_SIZE=230
MAX_ITERS=7000

MODELS=(
  ukb_amk125_clinical_icd
  ukb_amk125_clinical_demographics_icd
  ukb_amk125_clinical_demographics_ukb_icd
  ukb_amk125_clinical_demographics_ukb_biochem_icd
  ukb_amk125_clinical_demographics_ukb_biochem_icd_self_reported
)

CONFIGS=(
  config/train_clinical_icd.py
  config/train_clinical_demographics_icd.py
  config/train_clinical_demographics_ukb_icd.py
  config/train_clinical_demographics_ukb_biochem_icd.py
  config/train_ukb_amk125.py
)

ARRAY_INDEX=${MODEL_INDEX:-${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-}}}
if [[ -z "${ARRAY_INDEX}" || "${ARRAY_INDEX}" -lt 1 || "${ARRAY_INDEX}" -gt 5 ]]; then
  echo "ERROR: PBS array index must be between 1 and 5; got '${ARRAY_INDEX}'" >&2
  exit 2
fi

MODEL_NAME=${MODELS[$((ARRAY_INDEX - 1))]}
CONFIG=${CONFIGS[$((ARRAY_INDEX - 1))]}
DATA_DIR="${DATA_ROOT}/${MODEL_NAME}"

# Use a new directory so the existing block-size-128 checkpoints are preserved.
OUTPUT_DIR="models_bs230_7000/${MODEL_NAME}"
CHECKPOINT="${PROJECT_DIR}/${OUTPUT_DIR}/ckpt.pt"
RESULT_VERSION="${MODEL_NAME}_bs230"
EVALUATION_DIR="${PROJECT_DIR}/results/${RESULT_VERSION}/evaluation"

echo "Started:       $(date)"
echo "Host:          $(hostname)"
echo "PBS job:       ${PBS_JOBID:-not_set}"
echo "Array index:   ${ARRAY_INDEX}"
echo "Model:         ${MODEL_NAME}"
echo "Config:        ${CONFIG}"
echo "Data:          ${DATA_DIR}"
echo "Output:        ${PROJECT_DIR}/${OUTPUT_DIR}"
echo "Evaluation:    ${EVALUATION_DIR}"
echo "Block size:    ${BLOCK_SIZE}"
echo "Max iterations:${MAX_ITERS}"
echo "CPU cores:     ${NCPUS}"

module purge

[[ -x "${PYTHON}" ]] || { echo "ERROR: Python not found: ${PYTHON}" >&2; exit 3; }
[[ -f "${PROJECT_DIR}/train.py" ]] || { echo "ERROR: train.py not found" >&2; exit 3; }
[[ -f "${PROJECT_DIR}/${CONFIG}" ]] || { echo "ERROR: Config not found: ${CONFIG}" >&2; exit 3; }

for file in train.bin val.bin config_values.py; do
  [[ -s "${DATA_DIR}/${file}" ]] || {
    echo "ERROR: Missing or empty file: ${DATA_DIR}/${file}" >&2
    exit 3
  }
done

mkdir -p "${PROJECT_DIR}/${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

"${PYTHON}" -c \
  "import os, torch; torch.set_num_threads(int(os.environ['OMP_NUM_THREADS'])); print('PyTorch:', torch.__version__); print('Training device: CPU'); print('PyTorch CPU threads:', torch.get_num_threads())"

if [[ "${SKIP_TRAINING:-0}" == "1" ]]; then
  echo "SKIP_TRAINING=1: reusing checkpoint ${CHECKPOINT}"
else
  "${PYTHON}" train.py "${CONFIG}" \
    --out_dir="${OUTPUT_DIR}" \
    --block_size=${BLOCK_SIZE} \
    --max_iters=${MAX_ITERS} \
    --lr_decay_iters=${MAX_ITERS} \
    --device=cpu \
    --dtype=float32 \
    --compile=False

  echo "Training completed for ${MODEL_NAME}: $(date)"
fi

[[ -s "${CHECKPOINT}" ]] || {
  echo "ERROR: Training finished but checkpoint is missing: ${CHECKPOINT}" >&2
  exit 4
}
[[ -f "${DATA_DIR}/test.bin" ]] || {
  echo "ERROR: Test data not found: ${DATA_DIR}/test.bin" >&2
  exit 4
}
[[ -f "${DATA_DIR}/token_dictionary.csv" ]] || {
  echo "ERROR: Token dictionary not found: ${DATA_DIR}/token_dictionary.csv" >&2
  exit 4
}

mkdir -p "${EVALUATION_DIR}"

echo "Starting held-out test evaluation for ${MODEL_NAME}: $(date)"

# Evaluation uses the CPU while remaining inside the same array job. The
# context length is stated explicitly and must match the trained checkpoint.
"${PYTHON}" "${PROJECT_DIR}/run_evaluation.py" \
  --input_path "${DATA_DIR}" \
  --model_ckpt_path "${CHECKPOINT}" \
  --output_path "${EVALUATION_DIR}" \
  --model_name "${RESULT_VERSION}" \
  --split test \
  --block_size ${BLOCK_SIZE} \
  --device cpu \
  --batch_size 64 \
  --no_event_token_rate 5 \
  --count_threshold 1000 \
  --offset 365.25 \
  --dataset_subset_size -1 \
  --disease_chunk_size 512 \
  --analysis_patients 256 \
  --analysis_diseases 10 \
  --max_curve_plots 15 \
  --no-extended_analysis

REQUIRED_OUTPUTS=(
  auc_table.csv
  auc_results.csv
  auc_results_filtered.csv
  evaluation_summary.csv
  model_report.csv
  pr_curves.csv
  roc_curves.csv
  evaluation_run_config.csv
)

for output_file in "${REQUIRED_OUTPUTS[@]}"; do
  [[ -s "${EVALUATION_DIR}/${output_file}" ]] || {
    echo "ERROR: Evaluation did not create ${EVALUATION_DIR}/${output_file}" >&2
    exit 5
  }
done

touch "${EVALUATION_DIR}/EVALUATION_COMPLETE"
echo "Training and evaluation completed for ${MODEL_NAME}: $(date)"
