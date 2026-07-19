#!/bin/bash
#PBS -N delphi_dryrun
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1:gpu_type=A100
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o /rds/general/project/hda_24-25/live/amk125_thesis/logs/delphi_train_dryrun.log
echo "Starting Delphi dry run at $(date)"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

eval "$(~/anaconda3/bin/conda shell.bash hook)"
conda activate delphi

cd /rds/general/project/hda_24-25/live/amk125_thesis/Delphi/

# Run dryrun for each configuration so all four data folders are exercised
for CONFIG in \
    dryrun_clinical \
    dryrun_clinical_demographics \
    dryrun_clinical_demographics_ukb \
    dryrun_clinical_demographics_ukb_biochem
do
    echo "--- Training config: $CONFIG at $(date) ---"
    python train.py config/${CONFIG}.py \
        --device=cuda \
        --dtype=bfloat16 \
        --compile=False
    echo "--- Done: $CONFIG ---"
done

echo "All dry runs done at $(date)"