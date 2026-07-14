#!/bin/bash
#PBS -N delphi_train_full
#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1:gpu_type=A100
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /rds/general/project/hda_24-25/live/amk125_thesis/logs/delphi_train_full.log

echo "Starting Delphi full training at $(date)"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
eval "$(~/anaconda3/bin/conda shell.bash hook)"
conda activate delphi
cd /rds/general/project/hda_24-25/live/amk125_thesis/Delphi
python train.py config/train_delphi_amk125.py \
    --device=cuda \
    --dtype=bfloat16 \
    --compile=False
echo "Full training done at $(date)"