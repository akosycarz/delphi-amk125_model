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
python train.py config/ukb_amk125_dryrun.py \
    --device=cuda \
    --dtype=bfloat16 \
    --compile=False
echo "Dry run done at $(date)"