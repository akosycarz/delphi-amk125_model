#!/bin/bash
#PBS -N extract_clinical_history
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -o /rds/general/project/hda_24-25/live/amk125_thesis/scripts/logs/extract_clinical_history.stdout
#PBS -e /rds/general/project/hda_24-25/live/amk125_thesis/scripts/logs/extract_clinical_history.stderr

cd /rds/general/project/hda_24-25/live/amk125_thesis/scripts

eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate r413

Rscript 1-extract_clinical_history.R
