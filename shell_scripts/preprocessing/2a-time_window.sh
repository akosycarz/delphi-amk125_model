#!/bin/bash
#PBS -N time_window_analysis
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -o /rds/general/project/hda_24-25/live/amk125_thesis/scripts/logs/time_window_analysis.stdout
#PBS -e /rds/general/project/hda_24-25/live/amk125_thesis/scripts/logs/time_window_analysis.stderr

cd /rds/general/project/hda_24-25/live/amk125_thesis/scripts

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate r413

Rscript 2a-time_window_analysis.R
