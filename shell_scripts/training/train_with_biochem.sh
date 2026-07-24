#!/bin/bash
#PBS -N train_biochem
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=08:00:00
#PBS -j oe
#PBS -o /rds/general/user/amk125/ephemeral/logs/train_with_biochem.log

set -euo pipefail

PROJECT_DIR="/rds/general/user/amk125/home/delphi-amk125_model/Delphi"
DATA_DIR="/rds/general/user/amk125/home/delphi-amk125_model/data/ukb_amk125_clinical_demographics_ukb_biochem_icd"
PYTHON="/rds/general/user/amk125/home/anaconda3/envs/delphi/bin/python"
CONFIG="config/train_clinical_demographics_ukb_biochem_icd.py"

eval "$(~/anaconda3/bin/conda shell.bash hook)"
conda activate delphi

echo "Starting training with biochemistry at $(date)"
echo "Compute node: $(hostname)"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-not_set}"

module purge

[[ -x "${PYTHON}" ]] || { echo "ERROR: Missing Python: ${PYTHON}" >&2; exit 1; }
[[ -f "${PROJECT_DIR}/${CONFIG}" ]] || { echo "ERROR: Missing config: ${PROJECT_DIR}/${CONFIG}" >&2; exit 1; }

for file in train.bin val.bin config_values.py; do
  [[ -s "${DATA_DIR}/${file}" ]] || { echo "ERROR: Missing or empty: ${DATA_DIR}/${file}" >&2; exit 1; }
done

cd "${PROJECT_DIR}"

"${PYTHON}" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'PBS job did not receive a CUDA GPU'"

"${PYTHON}" train.py "${CONFIG}" --device=cuda --dtype=bfloat16 --compile=False

echo "Training with biochemistry completed successfully at $(date)"
