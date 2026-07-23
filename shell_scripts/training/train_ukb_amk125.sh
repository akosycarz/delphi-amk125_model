#!/bin/bash
#PBS -N train_ukb_base
#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /rds/general/user/amk125/ephemeral/logs/train_ukb_amk125.log

set -euo pipefail

PROJECT_DIR="${HOME}/delphi-amk125_model"
DATASET="ukb_amk125"
CONFIG="config/train_ukb_amk125.py"
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
