#!/bin/bash
#PBS -N train_clin_icd
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o /rds/general/user/amk125/ephemeral/logs/train_clinical_icd.log

set -euo pipefail

PROJECT_DIR="${HOME}/delphi-amk125_model"
DATASET="ukb_amk125_clinical_icd"
CONFIG="config/train_ukb_amk125_clinical_icd.py"
DATA_DIR="${PROJECT_DIR}/data/${DATASET}"

echo "Starting ${DATASET} training at $(date)"
echo "Node: $(hostname)"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-not_set}"

for file in train.bin val.bin config_values.py; do
  [[ -s "${DATA_DIR}/${file}" ]] || { echo "ERROR: Missing ${DATA_DIR}/${file}" >&2; exit 1; }
done

eval "$(~/anaconda3/bin/conda shell.bash hook)"
conda activate delphi
cd "${PROJECT_DIR}"

python train.py "${CONFIG}" --device=cuda --dtype=bfloat16 --compile=False

echo "Training completed successfully at $(date)"
