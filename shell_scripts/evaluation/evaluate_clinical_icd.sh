#!/bin/bash
#PBS -N eval_clin_icd
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -l walltime=08:00:00
#PBS -j oe
#PBS -o /rds/general/user/amk125/ephemeral/logs/evaluate_clinical_icd.log

set -euo pipefail

PROJECT_DIR="/rds/general/user/amk125/home/delphi-amk125_model/Delphi"
PYTHON="/rds/general/user/amk125/home/anaconda3/envs/delphi/bin/python"

module purge
cd "${PROJECT_DIR}"

"${PYTHON}" run_evaluation.py \
  --input_path /rds/general/user/amk125/home/delphi-amk125_model/data/ukb_amk125_clinical_icd \
  --model_ckpt_path "${PROJECT_DIR}/models/ukb_amk125_clinical_icd/ckpt.pt" \
  --output_path "${PROJECT_DIR}/models/ukb_amk125_clinical_icd/evaluation_test_retry" \
  --split test \
  --block_size 128 \
  --batch_size 16 \
  --no_event_token_rate 5 \
  --dataset_subset_size -1 \
  --device cpu

echo "Evaluation completed successfully at $(date)"
