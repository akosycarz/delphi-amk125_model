#!/bin/bash
#PBS -N delphi_dryrun
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o /rds/general/user/amk125/ephemeral/logs/delphi_dryrun.log

set -euo pipefail

PROJECT_DIR="/rds/general/user/amk125/home/delphi-amk125_model/Delphi"
DATA_DIR="/rds/general/user/amk125/home/delphi-amk125_model/data/ukb_amk125_clinical_icd"
PYTHON="/rds/general/user/amk125/home/anaconda3/envs/delphi/bin/python"
CONFIG="config/dryrun.py"

echo "Starting Delphi dry run at $(date)"
echo "Compute node: $(hostname)"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-not_set}"
echo "Project directory: ${PROJECT_DIR}"
echo "Dataset directory: ${DATA_DIR}"

module purge

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: Delphi Python executable not found: ${PYTHON}" >&2
  exit 1
fi

if [[ ! -f "${PROJECT_DIR}/${CONFIG}" ]]; then
  echo "ERROR: Dry-run config not found: ${PROJECT_DIR}/${CONFIG}" >&2
  exit 1
fi

for required_file in train.bin val.bin config_values.py; do
  if [[ ! -s "${DATA_DIR}/${required_file}" ]]; then
    echo "ERROR: Missing or empty file: ${DATA_DIR}/${required_file}" >&2
    exit 1
  fi
done

cd "${PROJECT_DIR}"

"${PYTHON}" -c \
  "import numpy, torch; print('NumPy:', numpy.__version__); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'PBS job did not receive a CUDA GPU'"

"${PYTHON}" train.py "${CONFIG}" \
  --device=cuda \
  --dtype=bfloat16 \
  --compile=False

echo "Delphi dry run completed successfully at $(date)"
